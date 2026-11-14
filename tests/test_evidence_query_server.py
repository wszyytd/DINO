"""Run ONLY on the authorized CUDA server with RUN_EVIDENCE_QUERY_CUDA_TESTS=1."""
import os
import unittest

ENABLED=os.environ.get('RUN_EVIDENCE_QUERY_CUDA_TESTS')=='1'
if ENABLED:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from dinoexp.evidence_memory import EvidenceMatcher
    from dinoexp.region_ab_prepare import cuda_device
    from dinoexp.region_sequence_training import advance
    from dinoexp.evidence_query_training import score_queries, query_losses, rollout_queries
    from dinoexp.evidence_query_eval import single_vehicle_control, repeat_control


@unittest.skipUnless(ENABLED,'CUDA authorization switch required; NOT verified locally')
class QueryServerTests(unittest.TestCase):
    def setUp(self):
        self.device=cuda_device('cuda:0')
        torch.manual_seed(7)
        self.model=EvidenceMatcher().to(self.device)
        b,t,k,j=1,3,2,3
        self.batch=dict(observation=torch.randn(b,t,k,16,256,device=self.device,requires_grad=True),
            observation_valid=torch.ones(b,t,k,16,device=self.device,dtype=torch.bool),
            step_valid=torch.ones(b,t,device=self.device,dtype=torch.bool),
            query=torch.randn(b,t,j,64,256,device=self.device),
            query_valid=torch.ones(b,t,j,64,device=self.device,dtype=torch.bool),
            query_label=torch.zeros(b,t,j,device=self.device,dtype=torch.long),
            query_active=torch.ones(b,t,j,device=self.device,dtype=torch.bool),
            query_group=torch.zeros(b,t,j,device=self.device,dtype=torch.long),
            repeat_candidate=torch.tensor([[-1,0,0]],device=self.device))
        self.cfg=dict(history_weight=1.,repeat_weight=.1)

    def read(self,t=0):
        b=self.batch
        return dict(queries=b['query'][:,t],valid=b['query_valid'][:,t],active=b['query_active'][:,t])

    def test_multiple_queries_equal_single_reads_and_state_unchanged(self):
        b=self.batch; s=advance(self.model,'memory',b['observation'][:,0],b['observation_valid'][:,0])
        before={k:v.clone() for k,v in s.items()}
        logits=score_queries(self.model,s,**self.read())
        for j in range(3):
            expected=self.model.score(b['query'][:,0,j],b['query_valid'][:,0,j],s)
            torch.testing.assert_close(logits[:,j],expected,atol=2e-5,rtol=0)
        for k in s: torch.testing.assert_close(s[k],before[k],atol=0,rtol=0)

    def test_loss_empty_groups_and_denominators(self):
        out=rollout_queries(self.model,'memory',self.batch,dict(self.cfg,repeat_weight=0))
        loss=query_losses(out['normal_logits'],None,out['batch'],self.cfg,torch.ones(2,device=self.device))
        self.assertEqual(loss['counts']['history_queries'],0)
        self.assertEqual(loss['counts']['current_queries'],9)
        self.assertTrue(torch.isfinite(loss['total']))
        batch={**out['batch'],'query_active':torch.zeros_like(self.batch['query_active'])}
        empty=query_losses(out['normal_logits'],None,batch,self.cfg,torch.ones(2,device=self.device))
        self.assertEqual(empty['total'].item(),0)
        # An inactive duplicate must not change the step/episode normalization.
        normal=out['normal_logits']; pad=torch.cat([normal,normal[:,:,:1]],2)
        extended=dict(out['batch'])
        for key in ('query_active','query_label','query_group'):
            extra=torch.zeros_like(extended[key][:,:,:1])
            extended[key]=torch.cat([extended[key],extra],2)
        got=query_losses(pad,None,extended,self.cfg,torch.ones(2,device=self.device))
        torch.testing.assert_close(loss['total'],got['total'])

    def test_repeat_target_detached(self):
        a=torch.tensor([[[[4.,0.,-1.]]]],device=self.device,requires_grad=True)
        r=torch.tensor([[[[1.,2.,0.]]]],device=self.device,requires_grad=True)
        b=dict(query_active=torch.ones(1,1,1,device=self.device,dtype=torch.bool),
            step_valid=torch.ones(1,1,device=self.device,dtype=torch.bool),
            query_label=torch.zeros(1,1,1,device=self.device,dtype=torch.long),
            query_group=torch.zeros(1,1,1,device=self.device,dtype=torch.long),
            repeat_active=torch.ones(1,1,device=self.device,dtype=torch.bool))
        loss=query_losses(a,r,b,self.cfg,torch.ones(2,device=self.device))
        loss['repeat'].backward()
        self.assertTrue(a.grad is None or a.grad.abs().sum()==0)
        self.assertGreater(r.grad.abs().sum().item(),0)

    def test_temporal_gradient_reaches_updater(self):
        out=rollout_queries(self.model,'memory',self.batch,self.cfg)
        # Final-step loss alone must reach the first observation and updater.
        out['normal_logits'][:,-1,:,0].sum().backward()
        self.assertGreater(self.batch['observation'].grad[:,0].abs().sum().item(),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.updater.parameters() if p.grad is not None),0)

    def test_repeat_branch_no_pollution_and_no_future_influence(self):
        a=rollout_queries(self.model,'memory',self.batch,self.cfg)
        b=rollout_queries(self.model,'memory',self.batch,dict(self.cfg,repeat_weight=0))
        torch.testing.assert_close(a['normal_logits'],b['normal_logits'],atol=0,rtol=0)
        for k in a['final_state']: torch.testing.assert_close(a['final_state'][k],b['final_state'][k],atol=0,rtol=0)
        changed=dict(self.batch); changed['observation']=self.batch['observation'].clone()
        changed['observation'][:,-1]+=20
        c=rollout_queries(self.model,'memory',changed,self.cfg)
        torch.testing.assert_close(a['normal_logits'][:,:-1],c['normal_logits'][:,:-1],atol=0,rtol=0)
        again=rollout_queries(self.model,'memory',self.batch,self.cfg)
        torch.testing.assert_close(a['normal_logits'],again['normal_logits'],atol=0,rtol=0)

    def test_absent_candidate_unchanged(self):
        b=self.batch
        for method in ('memory','mean','last'):
            from dinoexp.region_sequence_training import make_model
            m=make_model(dict(heads=4,slots=8),7,self.device,method)
            before=advance(m,method,b['observation'][:,0],b['observation_valid'][:,0])
            v=b['observation_valid'][:,1].clone(); v[:,0]=False
            after=advance(m,method,b['observation'][:,1],v,before)
            for key in before: torch.testing.assert_close(before[key][:,0],after[key][:,0],atol=0,rtol=0)

    def test_candidate_i_only_controls_leave_prefix_unchanged(self):
        b=self.batch; before=advance(self.model,'memory',b['observation'][:,0],b['observation_valid'][:,0])
        saved={k:v.clone() for k,v in before.items()}
        out=single_vehicle_control(self.model,'memory',before,b['observation'][:,1],b['observation_valid'][:,1],self.read(1),0)
        self.assertTrue(out['eligible'].all())
        self.assertEqual(out['skip_state']['count'][0,0].item(),1)
        self.assertEqual(out['skip_state']['count'][0,1].item(),2)
        torch.testing.assert_close(out['full_state']['seen'],out['skip_state']['seen'])
        repeat_control(self.model,'memory',out['full_state'],b['observation'][:,1],b['observation_valid'][:,1],self.read(1),0)
        for key in before: torch.testing.assert_close(before[key],saved[key],atol=0,rtol=0)

    def test_first_write_not_update_gain(self):
        b=self.batch
        empty=self.model.initial_state(1,2,self.device,b['observation'].dtype)
        out=single_vehicle_control(self.model,'memory',empty,b['observation'][:,0],b['observation_valid'][:,0],self.read(),0)
        self.assertFalse(out['eligible'].any()); self.assertTrue(out['first_write'].all())
        self.assertTrue(out['skip_state']['seen'][0,1])
        self.assertFalse(out['skip_state']['seen'][0,0])

    def test_candidate_permutation_and_query_change(self):
        original=rollout_queries(self.model,'memory',self.batch,self.cfg)
        changed=dict(self.batch); changed['observation']=self.batch['observation'][:,:,[1,0]]
        changed['observation_valid']=self.batch['observation_valid'][:,:,[1,0]]
        changed['repeat_candidate']=torch.where(self.batch['repeat_candidate']<0,-1,1-self.batch['repeat_candidate'])
        permuted=rollout_queries(self.model,'memory',changed,self.cfg)
        torch.testing.assert_close(original['normal_logits'],permuted['normal_logits'][...,[1,0,2]],atol=2e-5,rtol=0)
        changed=dict(self.batch); changed['query']=self.batch['query']+1
        q=rollout_queries(self.model,'memory',changed,self.cfg)
        for k in q['final_state']: torch.testing.assert_close(q['final_state'][k],original['final_state'][k],atol=0,rtol=0)

    def test_loss_hierarchical_hand_calculation(self):
        # Three valid steps over two episodes: equal episode weight, not flat CE.
        logits=torch.tensor([[[[2.,0.]],[[0.,2.]]],[[[0.,0.]],[[9.,-9.]]]],device=self.device,requires_grad=True)
        b=dict(query_active=torch.tensor([[[True],[True]],[[True],[False]]],device=self.device),
            step_valid=torch.tensor([[True,True],[True,False]],device=self.device),
            query_label=torch.zeros(2,2,1,device=self.device,dtype=torch.long),
            query_group=torch.zeros(2,2,1,device=self.device,dtype=torch.long))
        loss=query_losses(logits,None,b,self.cfg,torch.ones(2,device=self.device))
        ce=torch.nn.functional.cross_entropy(logits.reshape(-1,2),torch.zeros(4,device=self.device,dtype=torch.long),reduction='none')
        expected=((ce[0]+ce[1])/2+ce[2])/2
        torch.testing.assert_close(loss['current'],expected)
        self.assertEqual(loss['counts']['current_episodes'],2)
        self.assertEqual(loss['counts']['current_steps'],3)

    def test_inactive_query_never_passed_to_reader(self):
        b=self.batch; state=advance(self.model,'memory',b['observation'][:,0],b['observation_valid'][:,0])
        q=self.read(); q['active']=torch.zeros_like(q['active']); q['valid']=torch.zeros_like(q['valid'])
        logits=score_queries(self.model,state,**q)
        self.assertTrue(torch.isfinite(logits).all()); self.assertEqual(logits.abs().sum().item(),0.)

    def test_query_encoder_branch(self):
        # Branch contract using deterministic fake runtime; no pretrained download.
        import numpy as np
        from PIL import Image
        from dinoexp.evidence_query_prepare import encode_crop
        class Runtime:
            def independent(self,im): self.branch='independent'; return np.ones((37*37,256),np.float32)
            def reference(self,im,digest):
                self.branch='reference'; return None,None,np.ones((64,256),np.float32),np.ones(64,bool)
        rt=Runtime(); im=Image.new('RGB',(64,32))
        x,v=encode_crop(rt,im,'queries'); self.assertEqual(rt.branch,'reference'); self.assertEqual(x.shape,(64,256))
        x,v=encode_crop(rt,im,'observations'); self.assertEqual(rt.branch,'independent'); self.assertEqual(x.shape,(16,256))


if __name__=='__main__': unittest.main()
