"""Tests for the _CloudLogger class."""
import datetime
import threading
import time
from unittest import mock

from absl.testing import absltest
from ml_goodput_measurement.src import goodput

_CloudLogger = goodput._CloudLogger


class CloudLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-job'
    self.log_name = 'test-log'
    self.project_id = 'test-project-id'

  @mock.patch('google.cloud.logging.Client')
  def test_init_captures_project_id(self, mock_client_cls):
    """Test capture of project ID from the logging client."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    self.assertEqual(logger.project_id, self.project_id)
    self.assertEqual(logger.log_name, self.log_name)

  @mock.patch('google.cloud.logging.Client')
  def test_get_filter_msg(self, mock_client_cls):
    """Verifies the filter message construction for fast reads."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    filter_msg = logger._get_filter_msg(start_time=None, end_time=None)

    expected_full_log_name = f'projects/{self.project_id}/logs/{self.log_name}'
    self.assertIn(f'logName="{expected_full_log_name}"', filter_msg)
    self.assertIn(f'jsonPayload.job_name="{self.job_name}"', filter_msg)
    self.assertIn('severity=INFO', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_get_filter_msg_no_project(self, mock_client_cls):
    """Verifies safety: If project ID is missing, skip the optimization."""
    # Setup mock with no project ID.
    mock_instance = mock_client_cls.return_value
    mock_instance.project = None

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    filter_msg = logger._get_filter_msg(start_time=None, end_time=None)

    self.assertNotIn('logName=', filter_msg)
    self.assertIn(f'jsonPayload.job_name="{self.job_name}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_read_passes_filter_to_client(self, mock_client_cls):
    """Verifies that the filter is sent to the GCP client for fast reads."""
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.project = self.project_id
    mock_gcp_logger = mock_client_instance.logger.return_value

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    logger.read_cloud_logging_entries()

    mock_gcp_logger.list_entries.assert_called_once()

    _, kwargs = mock_gcp_logger.list_entries.call_args
    passed_filter = kwargs.get('filter_')
    self.assertIn(
        f'projects/{self.project_id}/logs/{self.log_name}', passed_filter
    )

  @mock.patch('google.cloud.logging.Client')
  def test_write_logs_entry(self, mock_client_cls):
    """Verifies entries are written to the logger."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value

    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    entry = {'job_name': self.job_name, 'data': 123}
    logger.write_cloud_logging_entry(entry)

    mock_gcp_logger.log_struct.assert_called_with(entry, severity='INFO')

  @mock.patch('google.cloud.logging.Client')
  def test_default_retention(self, mock_client_cls):
    """Verifies default retention period is applied to the filter."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)
    end_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_start = end_time - goodput._CLOUD_LOGGING_DEFAULT_RETENTION
    filter_msg = logger._get_filter_msg(start_time=None, end_time=end_time)
    self.assertIn(f'timestamp>"{expected_start.isoformat()}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_custom_retention(self, mock_client_cls):
    """Verifies custom retention period is applied to the filter."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    custom_retention = datetime.timedelta(hours=1)
    logger = goodput._CloudLogger(
        self.job_name, self.log_name, max_logs_retention_period=custom_retention
    )
    end_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_start = end_time - custom_retention
    filter_msg = logger._get_filter_msg(start_time=None, end_time=end_time)
    self.assertIn(f'timestamp>"{expected_start.isoformat()}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_calculator_passes_retention(self, mock_client_cls):
    """Verifies GoodputCalculator passes the retention down to _CloudLogger."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id
    custom_retention = datetime.timedelta(days=7)
    calculator = goodput.GoodputCalculator(
        self.job_name, self.log_name, max_logs_retention_period=custom_retention
    )
    self.assertEqual(
        calculator._cloud_logger.retention_period, custom_retention
    )


class CloudLoggerBackgroundWritesTest(absltest.TestCase):
  """Tests for the opt-in background-write path on `_CloudLogger`."""

  def setUp(self):
    super().setUp()
    self.job_name = 'test-job'
    self.log_name = 'test-log'

  @mock.patch('google.cloud.logging.Client')
  def test_default_keeps_sync_semantics(self, mock_client_cls):
    """Default constructor must keep historical synchronous behavior."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value
    mock_client_cls.return_value.project = 'p'

    logger = _CloudLogger(self.job_name, self.log_name)

    self.assertFalse(logger._background_writes_enabled)
    self.assertIsNone(logger._writer_thread)
    self.assertIsNone(logger._write_queue)

    entry = {goodput._JOB_NAME: self.job_name, 'data': 1}
    logger.write_cloud_logging_entry(entry)

    # Sync path: log_struct is called inline, with the original
    # 2-arg `(entry, severity='INFO')` shape — no `timestamp=` kwarg.
    mock_gcp_logger.log_struct.assert_called_once_with(entry, severity='INFO')

  @mock.patch('google.cloud.logging.Client')
  def test_background_writes_dispatch_off_caller_thread(self, mock_client_cls):
    """Background mode: caller returns immediately, daemon flushes later."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value
    mock_client_cls.return_value.project = 'p'

    caller_tid = threading.get_ident()
    flush_tids: list[int] = []

    def record_thread(*_args, **_kwargs):
      flush_tids.append(threading.get_ident())

    mock_gcp_logger.log_struct.side_effect = record_thread

    logger = _CloudLogger(
        self.job_name,
        self.log_name,
        enable_background_writes=True,
        background_flush_interval_s=0.05,
    )
    self.assertTrue(logger._background_writes_enabled)
    self.assertIsNotNone(logger._writer_thread)
    self.assertTrue(logger._writer_thread.is_alive())

    entry = {goodput._JOB_NAME: self.job_name, 'data': 1}
    logger.write_cloud_logging_entry(entry)

    # Caller thread should NOT have flushed inline.
    self.assertEqual(mock_gcp_logger.log_struct.call_count, 0)

    # Wait for the daemon to flush.
    deadline = time.time() + 2.0
    while time.time() < deadline and mock_gcp_logger.log_struct.call_count == 0:
      time.sleep(0.01)

    self.assertEqual(mock_gcp_logger.log_struct.call_count, 1)
    self.assertEqual(len(flush_tids), 1)
    self.assertNotEqual(flush_tids[0], caller_tid,
                        'log_struct must be called on the writer thread, '
                        'not the caller thread')

    logger.flush()  # Clean shutdown so atexit doesn't double-fire.

  @mock.patch('google.cloud.logging.Client')
  def test_background_writes_preserve_call_timestamp(self, mock_client_cls):
    """The Cloud Logging entry timestamp must be call-time, not flush-time."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value
    mock_client_cls.return_value.project = 'p'

    logger = _CloudLogger(
        self.job_name,
        self.log_name,
        enable_background_writes=True,
        background_flush_interval_s=0.05,
    )

    entry = {goodput._JOB_NAME: self.job_name, 'data': 1}
    before = datetime.datetime.now(datetime.timezone.utc)
    logger.write_cloud_logging_entry(entry)
    after = datetime.datetime.now(datetime.timezone.utc)

    # Force a flush with extra slack to avoid raciness on slow CI hosts.
    time.sleep(0.2)
    logger.flush()

    self.assertEqual(mock_gcp_logger.log_struct.call_count, 1)
    args, kwargs = mock_gcp_logger.log_struct.call_args
    self.assertEqual(args[0], entry)
    self.assertEqual(kwargs.get('severity'), 'INFO')
    ts = kwargs.get('timestamp')
    self.assertIsNotNone(ts, 'log_struct must be called with timestamp= '
                             'so GoodputCalculator filtering stays accurate')
    self.assertGreaterEqual(ts, before)
    self.assertLessEqual(ts, after)

  @mock.patch('google.cloud.logging.Client')
  def test_flush_drains_pending_entries(self, mock_client_cls):
    """flush() must drain whatever is queued before returning."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value
    mock_client_cls.return_value.project = 'p'

    logger = _CloudLogger(
        self.job_name,
        self.log_name,
        enable_background_writes=True,
        # Long interval so the worker won't flush on its own before we ask.
        background_flush_interval_s=600.0,
    )
    for i in range(25):
      logger.write_cloud_logging_entry(
          {goodput._JOB_NAME: self.job_name, 'i': i}
      )

    self.assertLess(mock_gcp_logger.log_struct.call_count, 25,
                    'entries should still be queued before flush()')

    logger.flush()
    self.assertEqual(mock_gcp_logger.log_struct.call_count, 25)

  @mock.patch('google.cloud.logging.Client')
  def test_writer_survives_log_struct_errors(self, mock_client_cls):
    """A raised log_struct must not halt the writer thread."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value
    mock_client_cls.return_value.project = 'p'

    # First call raises, subsequent calls succeed.
    mock_gcp_logger.log_struct.side_effect = [
        RuntimeError('transient'), None, None,
    ]

    logger = _CloudLogger(
        self.job_name,
        self.log_name,
        enable_background_writes=True,
        background_flush_interval_s=600.0,
    )
    for i in range(3):
      logger.write_cloud_logging_entry(
          {goodput._JOB_NAME: self.job_name, 'i': i}
      )
    logger.flush()

    # All 3 entries attempted; first raised but worker continued.
    self.assertEqual(mock_gcp_logger.log_struct.call_count, 3)

  @mock.patch('google.cloud.logging.Client')
  def test_recorder_passes_flag_to_cloud_logger(self, mock_client_cls):
    """GoodputRecorder must propagate enable_background_writes downstream."""
    mock_client_cls.return_value.project = 'p'

    recorder = goodput.GoodputRecorder(
        self.job_name,
        self.log_name,
        logging_enabled=True,
        enable_background_writes=True,
    )

    self.assertTrue(recorder._cloud_logger._background_writes_enabled)
    recorder._cloud_logger.flush()


if __name__ == '__main__':
  absltest.main()
  absltest.main()
