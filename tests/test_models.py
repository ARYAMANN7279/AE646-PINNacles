"""
Tests for AE646 Darcy Flow project code (src/models.py, src/train.py,
src/evaluate.py, src/preprocess.py, src/generate_data.py).
"""
import numpy as np
import pytest
import torch

from models import MLPBaseline, FNO2d, SpectralConv2d, get_model, count_parameters
from train import rel_l2_loss, physical_rel_l2
from evaluate import rel_l2, mse
from preprocess import add_coordinates, downsample, normalize_data


class TestModels:
    """Test model architectures."""

    def test_mlp_forward(self):
        model = MLPBaseline(input_channels=3, output_channels=1, height=64, width=64,
                             hidden_dims=[64, 64])
        x = torch.randn(2, 64, 64, 3)
        y = model(x)
        assert y.shape == (2, 64, 64, 1)

    def test_fno_forward(self):
        model = FNO2d(input_channels=3, output_channels=1, width=32, modes=8, n_layers=2)
        x = torch.randn(2, 64, 64, 3)
        y = model(x)
        assert y.shape == (2, 64, 64, 1)

    def test_spectral_conv_shape(self):
        conv = SpectralConv2d(in_channels=8, out_channels=8, modes1=4, modes2=4)
        x = torch.randn(2, 16, 16, 8)
        y = conv(x)
        assert y.shape == x.shape

    def test_model_factory(self):
        mlp = get_model("mlp", input_channels=3, output_channels=1, height=64, width=64)
        assert isinstance(mlp, MLPBaseline)
        fno = get_model("fno", input_channels=3, output_channels=1, width=32, modes=8)
        assert isinstance(fno, FNO2d)

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            get_model("invalid")

    def test_parameter_count(self):
        model = MLPBaseline(input_channels=3, output_channels=1, height=64, width=64,
                             hidden_dims=[64, 64])
        assert count_parameters(model) > 0


class TestMetrics:
    """Test relative-L2 metric implementations used in training/evaluation."""

    def test_rel_l2_loss_shape_and_nonneg(self):
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)
        loss = rel_l2_loss(pred, target)
        assert loss.shape == (4,)
        assert (loss >= 0).all()

    def test_rel_l2_zero_when_equal(self):
        x = torch.randn(3, 16, 16, 1)
        assert torch.allclose(rel_l2_loss(x, x), torch.zeros(3), atol=1e-6)

    def test_physical_rel_l2_invariant_to_standardization(self):
        """
        Denormalizing before computing relative error should reproduce the
        error computed directly in physical units (this guards against the
        original normalized-space metric bug: rel-L2 computed on standardized
        fields is NOT the same number as rel-L2 in physical units, because
        subtracting a constant mean changes ||target|| but not ||pred-target||).
        """
        torch.manual_seed(0)
        target_phys = torch.rand(5, 8, 8, 1) * 10 + 3.0
        pred_phys = target_phys + torch.randn(5, 8, 8, 1) * 0.5

        mean, std = target_phys.mean().item(), target_phys.std().item()
        target_norm = (target_phys - mean) / std
        pred_norm = (pred_phys - mean) / std

        expected = rel_l2_loss(pred_phys, target_phys)
        actual = physical_rel_l2(pred_norm, target_norm, mean, std)
        assert torch.allclose(expected, actual, atol=1e-5)

        # and it must differ from the (wrong) normalized-space error in general
        wrong = rel_l2_loss(pred_norm, target_norm)
        assert not torch.allclose(expected, wrong, atol=1e-3)

    def test_evaluate_rel_l2_and_mse(self):
        pred = torch.randn(4, 16, 16, 1)
        target = torch.randn(4, 16, 16, 1)
        err = rel_l2(pred, target)
        assert err.shape == (4,)
        assert (err >= 0).all()

        m = mse(pred, target)
        assert m.shape == (4,)
        assert (m >= 0).all()


class TestPreprocessing:
    """Test preprocessing utilities against small synthetic arrays."""

    def test_downsample_stride2(self):
        field = np.arange(16).reshape(1, 4, 4).astype(np.float32)
        ds = downsample(field, stride=2)
        assert ds.shape == (1, 2, 2)
        np.testing.assert_array_equal(ds[0], field[0][::2, ::2])

    def test_add_coordinates_shapes(self):
        coeff = np.random.randn(3, 8, 8).astype(np.float32)
        tensor = np.random.randn(3, 8, 8).astype(np.float32)
        x = np.linspace(0, 1, 8, dtype=np.float32)
        y = np.linspace(0, 1, 8, dtype=np.float32)
        inputs, targets = add_coordinates(coeff, tensor, x, y)
        assert inputs.shape == (3, 8, 8, 3)
        assert targets.shape == (3, 8, 8, 1)
        np.testing.assert_array_equal(inputs[..., 0], coeff)

    def test_normalize_uses_train_stats_only(self):
        rng = np.random.default_rng(0)
        train_c, val_c, test_c = rng.random((5, 4, 4)), rng.random((2, 4, 4)), rng.random((2, 4, 4))
        train_t, val_t, test_t = rng.random((5, 4, 4)), rng.random((2, 4, 4)), rng.random((2, 4, 4))

        (tr_c, tr_t, va_c, va_t, te_c, te_t, stats) = normalize_data(
            train_c, train_t, val_c, val_t, test_c, test_t
        )
        assert stats["coeff_mean"] == pytest.approx(train_c.mean())
        assert stats["tensor_mean"] == pytest.approx(train_t.mean())
        # train split itself should end up ~zero-mean/unit-std after normalization
        assert abs(tr_c.mean()) < 1e-5
        assert abs(tr_t.mean()) < 1e-5


class TestDataGeneration:
    """Test the optional synthetic-fallback data generator."""

    def test_permeability_is_piecewise_constant(self):
        from generate_data import generate_permeability_field, LOW_PERM, HIGH_PERM
        fields = generate_permeability_field(n_samples=2, height=16, width=16)
        assert fields.shape == (2, 16, 16)
        uniq = np.unique(fields)
        assert len(uniq) <= 2
        assert np.allclose(np.sort(uniq), sorted([LOW_PERM, HIGH_PERM])[:len(uniq)], atol=1e-6)

    def test_fdm_solver_shape_and_dirichlet_bc(self):
        from generate_data import solve_darcy_fdm
        coeff = np.ones((16, 16), dtype=np.float32)
        pressure = solve_darcy_fdm(coeff, height=16, width=16)
        assert pressure.shape == (16, 16)
        # Dirichlet BC: boundary should be (numerically) zero
        boundary = np.concatenate([pressure[0], pressure[-1], pressure[:, 0], pressure[:, -1]])
        assert np.allclose(boundary, 0, atol=1e-10)
        # interior of a constant-permeability field under a positive source should be positive
        assert pressure[8, 8] > 0


class TestIntegration:
    """Minimal end-to-end train/eval loop, no real data needed."""

    def test_train_eval_loop(self):
        torch.manual_seed(42)
        device = torch.device("cpu")

        model = FNO2d(input_channels=3, output_channels=1, width=16, modes=4, n_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        train_data = torch.utils.data.TensorDataset(
            torch.randn(8, 16, 16, 3), torch.randn(8, 16, 16, 1)
        )
        loader = torch.utils.data.DataLoader(train_data, batch_size=4)

        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            break

        model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                err = rel_l2_loss(outputs, targets)
                assert err.shape == (inputs.shape[0],)
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
