"""Copy reads between POD5 files with escapepod (shared by the data scripts).

escapepod read records reference their run info by index into the source
file's run-info table, so copying a read into a new file means adding its run
info to the writer (once per acquisition) and remapping that index.
"""

import escapepod

# ReadData fields passed straight through to Writer.add_read
_READ_FIELDS = (
    "read_id",
    "read_number",
    "start_sample",
    "channel",
    "well",
    "pore_type",
    "calibration_offset",
    "calibration_scale",
    "median_before",
    "end_reason",
    "end_reason_forced",
    "num_minknow_events",
    "tracked_scaling_scale",
    "tracked_scaling_shift",
    "predicted_scaling_scale",
    "predicted_scaling_shift",
    "num_reads_since_mux_change",
    "time_since_mux_change",
    "open_pore_level",
    "expected_open_pore_level",
    "selected_read_level",
)


class ReadCopier:
    """Copy selected reads from one or more readers into a single writer."""

    def __init__(self, writer: escapepod.Writer):
        self._writer = writer
        self._run_info_index: dict[str, int] = {}  # acquisition_id -> index

    def _map_run_info(self, reader: escapepod.Reader, index: int) -> int:
        run_info = reader.run_infos[index]
        key = run_info.acquisition_id
        if key not in self._run_info_index:
            self._run_info_index[key] = self._writer.add_run_info(run_info)
        return self._run_info_index[key]

    def copy(self, reader: escapepod.Reader, record: escapepod.ReadData) -> None:
        """Write one read (metadata + raw signal) from ``reader``."""
        fields = {name: getattr(record, name) for name in _READ_FIELDS}
        self._writer.add_read(
            **fields,
            run_info_index=self._map_run_info(reader, record.run_info_index),
            signal=reader.get_signal(record),
        )
