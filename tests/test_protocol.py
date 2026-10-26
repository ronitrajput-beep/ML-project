import unittest
import numpy as np
from sklearn.metrics import f1_score
from src.study import grid_f1,choose_threshold,augment,outage,environments,rank_groups,GRID
class ProtocolTests(unittest.TestCase):
 def test_threshold_math_matches_sklearn(self):
  rng=np.random.default_rng(1);y=rng.integers(0,2,40);p=rng.random(40)
  actual=grid_f1(y,p);expected=[f1_score(y,p>=t,zero_division=0) for t in GRID]
  np.testing.assert_allclose(actual,expected)
 def test_threshold_uses_equal_environment_scores(self):
  y=np.array([0,0,1,1]);ps={'clean':np.array([.1,.2,.8,.9]),'outage':np.array([.01,.02,.03,.04])}
  t,score=choose_threshold(y,ps,True)
  expected=np.mean([grid_f1(y,p) for p in ps.values()],axis=0)
  self.assertAlmostEqual(score,max(expected));self.assertAlmostEqual(t,GRID[np.flatnonzero(np.isclose(expected,max(expected),rtol=0,atol=1e-12))[-1]])
 def test_outage_never_changes_labels_or_source_array(self):
  x=np.arange(30,dtype=float).reshape(10,3);z=outage(x,[1]);self.assertTrue(np.isnan(z[:,1]).all());self.assertFalse(np.isnan(x).any());np.testing.assert_array_equal(x[:,0],z[:,0])
 def test_group_augmentation_preserves_original_weight(self):
  x=np.arange(60,dtype=float).reshape(10,6);y=np.array([0]*8+[1]*2);g={'a':[0,1],'b':[2],'c':[3]}
  xa,ya,w=augment(x,y,g,['a','b','c'],'group',1)
  self.assertEqual(xa.shape,(40,6));np.testing.assert_array_equal(ya,np.tile(y,4));np.testing.assert_allclose(w.reshape(4,10).sum(0),np.ones(10))
  self.assertTrue(np.isnan(xa[10:20,:2]).all());self.assertFalse(np.isnan(xa[:10]).any())
 def test_cell_control_matches_selected_cell_budget(self):
  x=np.ones((30,6));y=np.arange(30)%2;g={'a':[0,1],'b':[2],'c':[3]}
  xa,_,_=augment(x,y,g,['a','b','c'],'cell',2)
  self.assertEqual(np.isnan(xa[30:60]).sum(),60);self.assertEqual(np.isnan(xa[60:90]).sum(),30)
 def test_validation_cannot_see_held_out_group_masks(self):
  x=np.ones((10,5));g={str(i):[i] for i in range(5)}
  env=environments(x,g,list(g),1,False);self.assertEqual(list(env),['clean','group_1','group_2','group_3'])
  for z in env.values():self.assertFalse(np.isnan(z[:,3:]).any())
 def test_ranking_deterministic_from_train_only(self):
  x=np.array([[0.,1],[0,2],[10,1],[10,2]]);y=np.array([0,0,1,1]);g={'a':[0],'b':[1]}
  self.assertEqual(rank_groups(x,y,g),['a','b'])
if __name__=='__main__':unittest.main()
