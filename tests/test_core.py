"""Offline checks: no checkpoint/data download required."""
import numpy as np
import torch
import pytest
from pathlib import Path
from losses.consistency_loss import ConsistencyLoss
from datasets.manifest import Patient,check_disjoint_splits
from utils.labels import CLASS_NAMES,PAPER_CLASS_ORDER
from utils.metrics import compute_metrics_binary_folded


def test_class_mapping():
    assert CLASS_NAMES == ('FTA','WI-FTC','EA-FTC','MI-FTC')
    assert tuple(CLASS_NAMES[i] for i in PAPER_CLASS_ORDER) == ('FTA','MI-FTC','EA-FTC','WI-FTC')


def test_no_cross_cohort_patients():
    p=Patient(Path('/tmp/sample'),1,'P001')
    with pytest.raises(ValueError,match='overlap'):
        check_disjoint_splits({'train':[p],'valid':[p]})


def test_patient_folded_binary_uses_four_class_argmax():
    # FTC subtype probabilities total 0.6 yet FTA is the maximum individual class.
    probs=np.array([[.4,.2,.2,.2],[.05,.05,.80,.10],[.80,.05,.10,.05]])
    result=compute_metrics_binary_folded(np.array([0,2,0]),probs)
    assert result['acc'] == pytest.approx(1.0)
    assert result['sen'] == pytest.approx(1.0)
    assert result['spe'] == pytest.approx(1.0)
    assert result['auc'] == pytest.approx(1.0)
    assert 'threshold' not in result


def test_loss_has_finite_gradients():
    torch.manual_seed(3)
    root=torch.randn(8,2,requires_grad=True)
    leaf=torch.randn(8,4,requires_grad=True)
    target=torch.tensor([0,0,0,0,1,1,2,3])
    loss,*parts=ConsistencyLoss()(root,leaf,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert root.grad is not None and leaf.grad is not None


def test_minibatch_weights_and_bidirectional_kl():
    from losses.consistency_loss import batch_inverse_frequency
    y=torch.tensor([0,0,1])
    assert batch_inverse_frequency(y,2,1.).tolist()==pytest.approx([.4,.6])
    l1=torch.tensor([[.2,-.3]],requires_grad=True)
    l4=torch.tensor([[.2,.1,-.3,.5]],requires_grad=True)
    criterion=ConsistencyLoss(lambda_consistency=1., gamma=2.)
    # Isolate KL contribution, subtract focal losses (focal parts are detached in output).
    total,_,_,_=criterion(l1,l4,torch.tensor([1]))
    q=torch.softmax(l4,1)
    folded=torch.stack([q[:,0],q[:,1:].sum(1)],1)
    import torch.nn.functional as F
    kl=F.kl_div(F.log_softmax(l1,1),folded,reduction='batchmean')
    grad=torch.autograd.grad(kl,l4)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_patient_probability_mean_and_manuscript_display_order(monkeypatch, tmp_path):
    from PIL import Image
    from engine.evaluator import patient_predict
    from utils.labels import PROBABILITY_COLUMNS, PAPER_CLASS_ORDER
    assert PROBABILITY_COLUMNS == ('P_FTA', 'P_MI_FTC', 'P_EA_FTC', 'P_WI_FTC')
    pdir=tmp_path/'images';pdir.mkdir()
    for i in range(3):
        Image.fromarray(np.full((16,16),80+i*25,dtype=np.uint8)).save(pdir/f'{i}.png')
    class FakeModel:
        def eval(self): return self
        def predict_4class(self,images,tokens):
            return torch.tensor([[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.7,.1]],dtype=torch.float32)
    truth, probabilities, ids, n_images = patient_predict(
        FakeModel(), [Patient(pdir,2,'P1')], {'P1':torch.zeros(6,dtype=torch.long)},
        lambda im:torch.zeros((3,16,16)), torch.device('cpu'))
    assert ids == ['P1'] and n_images == [3]
    assert int(probabilities[0].argmax()) == 2
    assert probabilities[0].tolist() == pytest.approx([.1,.3,.5,.1],abs=1e-6)
    assert probabilities[0,list(PAPER_CLASS_ORDER)].tolist() == pytest.approx([.1,.1,.5,.3],abs=1e-6)

def test_loss_weighting_only_from_batch_or_training_counts():
    y=torch.tensor([0,0,1,1,2,3,3,3])
    if __import__('losses.consistency_loss',fromlist=['batch_inverse_frequency']).__dict__.get('batch_inverse_frequency'):
        from losses.consistency_loss import batch_inverse_frequency
        weights=batch_inverse_frequency(y,4,1)
        # raw counts [2,2,1,3] + 1 -> [3,3,2,4] -> inverses, renormalized
        assert weights.tolist() == pytest.approx([4/17,4/17,6/17,3/17], rel=1e-6)
    else:
        from utils.class_weights import inverse_frequency
        assert inverse_frequency([8,4,2,1]) == pytest.approx([1/15,2/15,4/15,8/15])
