"""Ingestion job store transitions and terminal-state helpers."""

from app.services import ingestion_job_service as jobs


def test_job_lifecycle_transitions():
    job = jobs.IngestionJob("doc.pdf")
    assert job.status == jobs.PENDING

    job.mark_processing()
    assert job.status == jobs.PROCESSING

    job.mark_completed(3)
    assert job.status == jobs.COMPLETED
    assert job.chunks_added == 3
    assert job.error is None


def test_failed_job_records_error():
    job = jobs.IngestionJob("bad.pdf")
    job.mark_failed("boom")
    assert job.status == jobs.FAILED
    assert job.error == "boom"


def test_is_terminal():
    assert jobs.is_terminal(jobs.COMPLETED) is True
    assert jobs.is_terminal(jobs.FAILED) is True
    assert jobs.is_terminal(jobs.PENDING) is False
    assert jobs.is_terminal(jobs.PROCESSING) is False


def test_store_creates_gets_and_lists():
    store = jobs.IngestionJobStore()
    job = store.create("x.pdf")
    assert store.get(job.job_id) is job
    assert store.get("missing") is None
    listed = store.list()
    assert listed[0]["job_id"] == job.job_id
    assert listed[0]["filename"] == "x.pdf"
    assert listed[0]["status"] == jobs.PENDING
