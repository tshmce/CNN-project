from src.models import efficientnet, resnet50, vgg16, vgg19


def _assert_model_output(model, num_classes):
    assert model.output_shape[-1] == num_classes


def test_vgg16_builder():
    model = vgg16.build_model(num_classes=3, weights=None)
    _assert_model_output(model, 3)


def test_vgg19_builder():
    model = vgg19.build_model(num_classes=3, weights=None)
    _assert_model_output(model, 3)


def test_resnet50_builder():
    model = resnet50.build_model(num_classes=3, weights=None)
    _assert_model_output(model, 3)


def test_efficientnet_builder():
    model = efficientnet.build_model(num_classes=3, weights=None)
    _assert_model_output(model, 3)
