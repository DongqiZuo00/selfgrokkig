"""Synthetic specification fixtures ONLY; never imported by a training runner.
Class names/assignments, paired intervals and checkpoint profiles below are
test inputs, not frozen production definitions or research results.
Per-stage Delta intentionally has NO default or implementation here.
"""
import math
import unittest

def explicit_class_score(passes, classes):
    if not passes or not classes or any(type(p) is not int or p not in (0,1) for p in passes):
        raise ValueError("Need nonempty binary per-test observations and explicit classes")
    indices=[i for members in classes.values() for i in members]
    if any(not members for members in classes.values()) or sorted(indices)!=list(range(len(passes))):
        raise ValueError("Fixture partition must cover every test exactly once with no empty class")
    fractions={name:sum(passes[i] for i in members)/len(members) for name,members in classes.items()}
    f=min(fractions.values())
    return fractions,f,int(f==1)

def course_credit(rates, successes, base_rung, q, paired_intervals):
    if q<=0 or len(rates)!=len(successes):raise ValueError("Invalid fixture")
    k=len(rates[0])-1 if sum(n>=q for n in successes)>=2 else base_rung
    gains=[r[k]-rates[0][k] for r in rates[1:]]
    positive=[i+1 for i,(g,(lo,hi)) in enumerate(zip(gains,paired_intervals)) if g>0 and lo>0 and hi>=lo]
    return k,gains,positive

def insert_fixture(archive,key,candidate,interval):
    if key not in archive or (candidate["rates"][key[0]]>archive[key]["rates"][key[0]] and interval[0]>0):
        archive[key]=candidate
        return True
    return False

def lead_fixture(archive):
    # Only unsaturated profiles and non-tied inputs are supplied in fixtures.
    return max(archive.items(),key=lambda kv:(kv[0][0],kv[1]["rates"][kv[0][0]]))[1]

class ExplicitSpecFixtures(unittest.TestCase):
    def test_minimum_is_not_pooled_fraction(self):
        fractions,f,full=explicit_class_score([1,1,1,0],{"FIXTURE_A":[0,1,2],"FIXTURE_B":[3]})
        self.assertEqual(f,0);self.assertEqual(full,0)
        self.assertNotEqual(f,3/4)
    def test_fixed_denominator_preserves_timeout_failure(self):
        fractions,f,full=explicit_class_score([0,1,1,1],{"FIXTURE_A":[0,1],"FIXTURE_B":[2,3]})
        self.assertEqual(fractions,{"FIXTURE_A":.5,"FIXTURE_B":1.})
        self.assertEqual((f,full),(.5,0))
    def test_empty_missing_duplicate_classes_rejected(self):
        for classes in [{},{"A":[],"B":[0,1]},{"A":[0]},{"A":[0],"B":[0,1]}]:
            with self.assertRaises(ValueError):explicit_class_score([1,1],classes)
    def test_full_pass_requires_all_classes(self):
        self.assertEqual(explicit_class_score([1,1],{"A":[0],"B":[1]})[1:],(1.,1))
    def test_switch_requires_two_branches_including_direct(self):
        r=[[1.,.2,0.],[1.,.4,.1],[1.,.3,0.]]
        self.assertEqual(course_credit(r,[0,4,0],1,4,[(.1,.3),(-.1,.2)])[0],1)
        k,g,j=course_credit(r,[4,4,0],1,4,[(.02,.18),(-.1,.1)])
        self.assertEqual(k,2);self.assertEqual(g,[.1,0.]);self.assertEqual(j,[1])
    def test_jplus_rejects_zero_or_crossing_ci(self):
        rates=[[1.,0.],[1.,.1],[1.,.1],[1.,0.]]
        self.assertEqual(course_credit(rates,[0]*4,1,4,[(0.,.2),(-.1,.3),(0.,0.)])[2],[])
    def test_archive_keeps_new_cells_and_requires_significant_replacement(self):
        a={};old={"rates":[1.,.6,0.]};new={"rates":[1.,.5,0.]}
        self.assertTrue(insert_fixture(a,(1,0),old,(0.,0.)))
        self.assertTrue(insert_fixture(a,(1,1),new,(-.2,-.1)))
        self.assertFalse(insert_fixture(a,(1,0),new,(-.2,-.1)))
        self.assertFalse(insert_fixture(a,(1,0),{"rates":[1.,.7,0.]},(-.1,.2)))
        self.assertTrue(insert_fixture(a,(1,0),{"rates":[1.,.7,0.]},(.01,.2)))
        self.assertEqual(len(a),2)
    def test_positive_gain_can_fail_to_change_archive(self):
        incumbent={"rates":[1.,.6,0.]};candidate={"rates":[1.,.5,0.]}
        a={(1,0):incumbent}
        k,g,j=course_credit([[1.,.4,0.],candidate["rates"]],[0,0],1,4,[(.05,.15)])
        self.assertEqual(j,[1])
        self.assertFalse(insert_fixture(a,(1,0),candidate,(-.15,-.05)))
    def test_lead_can_have_worse_full_pass(self):
        a={"name":"A","rates":[.99,.98,.97,.95,.88,.87]}
        b={"name":"B","rates":[.99,.98,.97,.95,.89,.01]}
        self.assertEqual(lead_fixture({(4,0):a,(4,1):b})["name"],"B")
    def test_hint_rate_increase_without_mixed_groups(self):
        groups=[[1]*8 for _ in range(8)]+[[0]*8 for _ in range(24)]
        self.assertGreater(sum(map(sum,groups))/(32*8),0.)
        self.assertEqual(sum(len(set(g))>1 for g in groups),0)
if __name__=="__main__":unittest.main(verbosity=2)

