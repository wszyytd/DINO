import unittest
import torch
from PIL import Image
from dinoexp.evidence_data import mask_patch_weights, pool_masked_regions
from dinoexp.evidence_training import partition_scenes, make_routes, pack, score_batch
from dinoexp.evidence_memory import EvidenceMatcher
from dinoexp.search_grid import build_graph


class EvidenceDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def scene(self):
        graph = build_graph(5, 4, [0, 0, 8], -45, move_speed=2, turn_speed=90, hold_s=.5)
        ids = list(graph['states'])
        return dict(graph=graph, state_ids=ids, image_hashes=['h'+str(i) for i in range(len(ids))],
                    objects=['A','B'], split='train', group='g', scene_id='s',
                    observations=torch.randn(len(ids), 2, 4, 16),
                    valid=torch.ones(len(ids), 2, 4, dtype=torch.bool),
                    references=torch.randn(2, 4, 16), reference_valid=torch.ones(2,4,dtype=torch.bool))

    def test_letterbox_mask_and_region_pooling(self):
        weights = mask_patch_weights(Image.new('L', (1024,768), 255))
        self.assertEqual(weights.shape, (37,37))
        self.assertEqual(float(weights[0].sum()), 0)
        self.assertAlmostEqual(float(weights.sum()) * 14**2, 518*388, places=1)
        features = torch.ones(37*37, 8)
        out, valid = pool_masked_regions(features, weights)
        torch.testing.assert_close(out[valid], torch.ones_like(out[valid]))
        out, valid = pool_masked_regions(features, weights*0)
        self.assertFalse(valid.any())
        self.assertFalse(out.any())

    def test_routes_stay_in_split_are_adjacent_and_pair_all_references(self):
        s = self.scene()
        meta=dict(teacher_training_images=['h30'], teacher_training_groups=['g'])
        parts, scope = partition_scenes([s],meta)
        fit, probe = set(parts['train'][0][1]),set(parts['probe'][0][1])
        self.assertFalse(fit & probe)
        self.assertIn('SINGLE-LAYOUT',scope)
        routes=make_routes([s],parts['probe'],count=4,length=8)
        self.assertEqual(len(routes),8)
        for route in routes:
            self.assertTrue(set(route['frames']) <= probe)
            for a,b,action in zip(route['frames'][:-1],route['frames'][1:],route['actions'], strict=True):
                self.assertEqual(s['graph']['states'][s['state_ids'][a]]['edges'][action]['to'],s['state_ids'][b])
        self.assertEqual(routes[0]['frames'],routes[1]['frames'])
        self.assertNotEqual(routes[0]['target'],routes[1]['target'])

    def test_shared_rgb_across_partitions_is_rejected(self):
        s=self.scene();s['image_hashes'][-1]=s['image_hashes'][0]
        with self.assertRaisesRegex(ValueError,'shared source'):
            partition_scenes([s],dict(teacher_training_images=[],teacher_training_groups=[]))

    def test_history_labels_and_future_causality(self):
        s=self.scene();s['valid'][1:3,0]=False
        row=dict(scene=0,frames=[0,1,2],target=0)
        batch=pack([row],[s],torch.device('cpu'))
        m=EvidenceMatcher(dim=16,heads=2,slots=3)
        _, records=score_batch(m,batch)
        self.assertTrue(records[1]['historical'][0])
        self.assertFalse(records[1]['visible'][0])
        changed=list(batch);changed[0]=batch[0].clone();changed[0][:,2]=100
        _, other=score_batch(m,tuple(changed))
        for i in range(2):
            torch.testing.assert_close(records[i]['confidence'],other[i]['confidence'])


if __name__ == '__main__':
    unittest.main()
