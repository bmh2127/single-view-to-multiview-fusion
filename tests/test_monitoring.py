def test_step_start_loads_data(monitoring_run):
    """Test that the start step loads reference and production data."""
    data = monitoring_run["start"].task.data
    assert hasattr(data, "reference_data"), "Reference data not loaded"
    assert hasattr(data, "current_data"), "Current data not loaded"
    # Note: In mock testing, data might be empty but the attributes should exist


def test_step_compute_metrics_processes_data(monitoring_run):
    """Test that compute_metrics step processes and creates metric dataframes."""
    data = monitoring_run["compute_metrics"].task.data
    assert hasattr(data, "reference_metrics_df"), "Reference metrics dataframe not created"
    assert hasattr(data, "current_metrics_df"), "Current metrics dataframe not created"


def test_step_performance_analysis_generates_html_report(monitoring_run):
    """Test that performance_analysis step generates an HTML report."""
    data = monitoring_run["performance_analysis"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated"
    assert isinstance(data.html, str), "HTML report is not a string"


def test_step_data_quality_report_generates_html_report(monitoring_run):
    """Test that data_quality_report step generates an HTML report."""
    data = monitoring_run["data_quality_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated"
    assert isinstance(data.html, str), "HTML report is not a string"


def test_step_n_ssim_test_generates_html_report(monitoring_run):
    """Test that n_ssim_test step generates an HTML report."""
    data = monitoring_run["n_ssim_test"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated"
    assert isinstance(data.html, str), "HTML report is not a string"


def test_step_visualize_samples_generates_html_report(monitoring_run):
    """Test that visualize_samples step generates an HTML report."""
    data = monitoring_run["visualize_samples"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated"
    assert isinstance(data.html, str), "HTML report is not a string"


def test_flow_completes_successfully(monitoring_run):
    """Test that the entire flow completes without errors."""
    assert monitoring_run["end"].finished_successfully(), "Flow did not complete successfully"