import torch

from cnn_segmentation import (
    SmallSegCNN,
    binary_metrics,
    segmentation_loss,
    trainable_parameter_count,
)


def test_small_seg_cnn_shape_and_size():
    model = SmallSegCNN()
    x = torch.randn(1, 3, 384, 288)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (1, 1, 384, 288)
    assert trainable_parameter_count(model) < 2_000_000


def test_loss_is_finite_and_backward_works():
    model = SmallSegCNN()
    x = torch.randn(1, 3, 64, 48)
    target = torch.zeros(1, 1, 64, 48)
    target[:, :, 12:52, 10:38] = 1.0
    logits = model(x)
    loss = segmentation_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_binary_metrics_perfect_and_empty_cases():
    target = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
    perfect = binary_metrics(target, target)
    assert perfect.dice == 1.0
    assert perfect.iou == 1.0
    empty = torch.zeros((2, 2), dtype=torch.uint8)
    empty_metrics = binary_metrics(empty, empty)
    assert empty_metrics.dice == 1.0
    assert empty_metrics.iou == 1.0
