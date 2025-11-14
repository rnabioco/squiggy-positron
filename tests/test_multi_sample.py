"""Tests for multi-sample functionality in SquiggyKernel"""


class TestSample:
    """Tests for Sample class"""

    def test_sample_initialization(self):
        """Test that Sample initializes correctly"""
        from squiggy.io import Sample

        sample = Sample("model_v4.2")

        assert sample.name == "model_v4.2"
        assert sample._pod5_path is None
        assert sample._pod5_reader is None
        assert sample._read_ids == []
        assert sample._bam_path is None
        assert sample._bam_info is None

    def test_sample_repr_empty(self):
        """Test Sample repr when empty"""
        from squiggy.io import Sample

        sample = Sample("test")
        assert "<Sample(test): No files loaded>" in repr(sample)

    def test_sample_repr_with_pod5(self, sample_pod5_file):
        """Test Sample repr with POD5 loaded"""
        import pod5

        from squiggy.io import Sample

        sample = Sample("model_v4.2")
        reader = pod5.Reader(str(sample_pod5_file))
        read_ids = [str(read.read_id) for read in reader.reads()]

        sample._pod5_path = str(sample_pod5_file)
        sample._pod5_reader = reader
        sample._read_ids = read_ids

        repr_str = repr(sample)
        assert "model_v4.2" in repr_str
        assert "reads" in repr_str

    def test_sample_close(self, sample_pod5_file):
        """Test that sample.close() properly cleans up"""
        import pod5

        from squiggy.io import Sample

        sample = Sample("test")
        reader = pod5.Reader(str(sample_pod5_file))
        read_ids = [str(read.read_id) for read in reader.reads()]

        sample._pod5_path = str(sample_pod5_file)
        sample._pod5_reader = reader
        sample._read_ids = read_ids

        sample.close()

        assert sample._pod5_reader is None
        assert sample._pod5_path is None
        assert sample._read_ids == []


class TestSquiggyKernelMultiSample:
    """Tests for multi-sample features of SquiggyKernel"""

    def testsquiggy_kernel_samples_dict(self):
        """Test that SquiggyKernel has samples dict"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        assert isinstance(session.samples, dict)
        assert len(session.samples) == 0

    def test_load_sample(self, sample_pod5_file, sample_bam_file):
        """Test loading a sample into session"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        sample = session.load_sample(
            "v4.2", str(sample_pod5_file), str(sample_bam_file)
        )

        assert sample.name == "v4.2"
        assert sample._pod5_path is not None
        assert sample._pod5_reader is not None
        assert len(sample._read_ids) > 0
        assert sample._bam_path is not None
        assert sample._bam_info is not None

        # Should be in samples dict
        assert "v4.2" in session.samples
        assert session.samples["v4.2"] is sample

    def test_load_multiple_samples(self, sample_pod5_file, sample_bam_file):
        """Test loading multiple samples"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()

        sample_a = session.load_sample(
            "v4.2", str(sample_pod5_file), str(sample_bam_file)
        )
        sample_b = session.load_sample(
            "v5.0", str(sample_pod5_file), str(sample_bam_file)
        )

        assert len(session.samples) == 2
        assert "v4.2" in session.samples
        assert "v5.0" in session.samples
        assert session.samples["v4.2"] is sample_a
        assert session.samples["v5.0"] is sample_b

    def test_get_sample(self, sample_pod5_file):
        """Test getting a sample by name"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        loaded_sample = session.load_sample("test", str(sample_pod5_file))

        retrieved_sample = session.get_sample("test")
        assert retrieved_sample is loaded_sample

        # Non-existent sample returns None
        assert session.get_sample("nonexistent") is None

    def test_list_samples(self, sample_pod5_file):
        """Test listing all loaded samples"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        assert session.list_samples() == []

        session.load_sample("v4.2", str(sample_pod5_file))
        session.load_sample("v5.0", str(sample_pod5_file))

        samples = session.list_samples()
        assert len(samples) == 2
        assert "v4.2" in samples
        assert "v5.0" in samples

    def test_remove_sample(self, sample_pod5_file):
        """Test removing a sample"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        session.load_sample("v4.2", str(sample_pod5_file))

        assert "v4.2" in session.samples
        session.remove_sample("v4.2")
        assert "v4.2" not in session.samples

    def test_remove_nonexistent_sample(self):
        """Test removing nonexistent sample doesn't raise error"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        session.remove_sample("nonexistent")  # Should not raise

    def test_close_all(self, sample_pod5_file):
        """Test closing all samples"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        session.load_sample("v4.2", str(sample_pod5_file))
        session.load_sample("v5.0", str(sample_pod5_file))

        assert len(session.samples) == 2
        session.close_all()
        assert len(session.samples) == 0

    def test_session_repr_multi_sample(self, sample_pod5_file):
        """Test session repr with multiple samples"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        session.load_sample("v4.2", str(sample_pod5_file))
        session.load_sample("v5.0", str(sample_pod5_file))

        repr_str = repr(session)
        assert "2 sample(s)" in repr_str
        assert "v4.2" in repr_str
        assert "v5.0" in repr_str


class TestPublicAPI:
    """Tests for public API functions"""

    def test_load_sample_function(self, sample_pod5_file):
        """Test load_sample() convenience function"""
        from squiggy import load_sample
        from squiggy.io import squiggy_kernel

        sample = load_sample("test", str(sample_pod5_file))

        assert sample.name == "test"
        assert "test" in squiggy_kernel.samples

    def test_get_sample_function(self, sample_pod5_file):
        """Test get_sample() convenience function"""
        from squiggy import get_sample, load_sample

        load_sample("test", str(sample_pod5_file))
        sample = get_sample("test")

        assert sample is not None
        assert sample.name == "test"

    def test_list_samples_function(self, sample_pod5_file):
        """Test list_samples() convenience function"""
        from squiggy import list_samples, load_sample

        load_sample("v4.2", str(sample_pod5_file))
        load_sample("v5.0", str(sample_pod5_file))

        samples = list_samples()
        assert len(samples) == 2
        assert "v4.2" in samples
        assert "v5.0" in samples

    def test_remove_sample_function(self, sample_pod5_file):
        """Test remove_sample() convenience function"""
        from squiggy import load_sample, remove_sample
        from squiggy.io import squiggy_kernel

        load_sample("test", str(sample_pod5_file))
        assert "test" in squiggy_kernel.samples

        remove_sample("test")
        assert "test" not in squiggy_kernel.samples

    def test_close_all_samples_function(self, sample_pod5_file):
        """Test close_all_samples() convenience function"""
        from squiggy import close_all_samples, load_sample
        from squiggy.io import squiggy_kernel

        load_sample("v4.2", str(sample_pod5_file))
        load_sample("v5.0", str(sample_pod5_file))
        assert len(squiggy_kernel.samples) == 2

        close_all_samples()
        assert len(squiggy_kernel.samples) == 0


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with single-sample API"""

    def test_single_sample_mode_still_works(self, sample_pod5_file):
        """Test that single-sample load_pod5() still works"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        load_pod5(str(sample_pod5_file))

        # Old API still works
        assert squiggy_kernel._pod5_path is not None
        assert len(squiggy_kernel._read_ids) > 0

    def test_mixing_old_and_new_api(self, sample_pod5_file, sample_bam_file):
        """Test mixing old single-sample and new multi-sample APIs"""
        from squiggy import load_bam, load_pod5, load_sample
        from squiggy.io import squiggy_kernel

        # Load single-sample (old API)
        load_pod5(str(sample_pod5_file))
        load_bam(str(sample_bam_file))

        # Also load named sample (new API)
        load_sample("named", str(sample_pod5_file), str(sample_bam_file))

        # Both should work
        assert squiggy_kernel._pod5_path is not None
        assert squiggy_kernel._bam_path is not None
        assert "named" in squiggy_kernel.samples


class TestSampleIntegration:
    """Integration tests for multi-sample workflow"""

    def test_comparison_workflow(self, sample_pod5_file, sample_bam_file):
        """Test typical A/B comparison workflow"""
        from squiggy import get_sample, list_samples, load_sample

        # Load two basecaller versions
        load_sample("dorado_v4.2", str(sample_pod5_file), str(sample_bam_file))
        load_sample("dorado_v5.0", str(sample_pod5_file), str(sample_bam_file))

        # List all samples
        samples = list_samples()
        assert len(samples) == 2

        # Access individual samples
        sample_a = get_sample("dorado_v4.2")
        sample_b = get_sample("dorado_v5.0")

        assert sample_a is not None
        assert sample_b is not None
        assert len(sample_a._read_ids) > 0
        assert len(sample_b._read_ids) > 0

    def test_replicate_samples(self, sample_pod5_file, sample_bam_file):
        """Test loading replicate samples"""
        from squiggy import list_samples, load_sample

        # Load technical replicates
        load_sample("replicate_1", str(sample_pod5_file), str(sample_bam_file))
        load_sample("replicate_2", str(sample_pod5_file), str(sample_bam_file))
        load_sample("replicate_3", str(sample_pod5_file), str(sample_bam_file))

        samples = list_samples()
        assert len(samples) == 3
        assert "replicate_1" in samples
        assert "replicate_2" in samples
        assert "replicate_3" in samples
