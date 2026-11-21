"""Unit and Integration tests for FastAPI OMR Interpreter API."""

import io
import json
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

from src.app import app, load_components
import config


@pytest.fixture(scope="module")
def client():
    load_components()
    with TestClient(app) as test_client:
        yield test_client


def create_synthetic_sheet_bytes() -> bytes:
    """Create a minimal synthetic sheet image for API testing."""
    h, w = 1754, 1240
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw corner fiducials/anchors
    anchors = [
        (40, 40, 60, 60),
        (w - 100, 40, 60, 60),
        (40, h - 100, 60, 60),
        (w - 100, h - 100, 60, 60),
    ]
    for x, y, aw, ah in anchors:
        cv2.rectangle(img, (x, y), (x + aw, y + ah), (0, 0, 0), -1)

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


def test_health_endpoint(client):
    """Test /api/health returns 200 and expected environment info."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "models" in data
    assert data["template_questions"] == 120


def test_serve_frontend_index(client):
    """Test GET / serves frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "OMR" in response.text
    assert "Answer Sheet Interpreter" in response.text


def test_grade_invalid_file_type(client):
    """Test POST /api/grade rejects unsupported file formats."""
    fake_txt = io.BytesIO(b"Not an image")
    response = client.post(
        "/api/grade",
        files={"image": ("document.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_grade_file_too_large(client):
    """Test POST /api/grade rejects files larger than 25MB."""
    # Create payload slightly exceeding 25MB
    large_payload = b"0" * (26 * 1024 * 1024)
    response = client.post(
        "/api/grade",
        files={"image": ("large.jpg", io.BytesIO(large_payload), "image/jpeg")}
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]


def test_grade_synthetic_sheet_end_to_end(client):
    """Test full end-to-end grading on a synthetic sheet image."""
    sheet_bytes = create_synthetic_sheet_bytes()
    response = client.post(
        "/api/grade",
        files={"image": ("sheet.jpg", io.BytesIO(sheet_bytes), "image/jpeg")},
        data={"marks_per_correct": "1.0", "negative_marks": "0.25", "multi_mark_marks": "0.0"}
    )
    assert response.status_code == 200
    data = response.json()

    assert "result_id" in data
    assert "summary" in data
    assert "student_details" in data
    assert "questions" in data
    assert "flagged_issues" in data
    assert "overlay_preview" in data

    # 120 questions evaluated
    assert len(data["questions"]) == 120
    assert data["summary"]["total_marks"] == 120.0

    result_id = data["result_id"]

    # Test GET /api/results/{result_id}
    res_resp = client.get(f"/api/results/{result_id}")
    assert res_resp.status_code == 200
    assert res_resp.json()["result_id"] == result_id

    # Test GET /api/export/{result_id}.csv
    csv_resp = client.get(f"/api/export/{result_id}.csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "Question,Selected Option(s),Expected Answer,Status,Marks Awarded" in csv_resp.text

    # Test GET /api/export/{result_id}.json
    json_resp = client.get(f"/api/export/{result_id}.json")
    assert json_resp.status_code == 200
    assert "application/json" in json_resp.headers["content-type"]
    exported_data = json_resp.json()
    assert "summary" in exported_data
    assert "questions" in exported_data


def test_grade_with_custom_json_answer_key(client):
    """Test grading with an uploaded custom JSON answer key."""
    sheet_bytes = create_synthetic_sheet_bytes()
    custom_key = {"answers": {"1": "A", "2": "B", "3": "C"}}
    key_bytes = json.dumps(custom_key).encode("utf-8")

    response = client.post(
        "/api/grade",
        files={
            "image": ("sheet.jpg", io.BytesIO(sheet_bytes), "image/jpeg"),
            "answer_key_file": ("custom_key.json", io.BytesIO(key_bytes), "application/json")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["questions"][0]["expected_answer"] == "A"
    assert data["questions"][1]["expected_answer"] == "B"
    assert data["questions"][2]["expected_answer"] == "C"


def test_grade_with_custom_csv_answer_key(client):
    """Test grading with an uploaded custom CSV answer key."""
    sheet_bytes = create_synthetic_sheet_bytes()
    csv_text = "Question,Answer\n1,D\n2,E\n3,A\n"
    csv_bytes = csv_text.encode("utf-8")

    response = client.post(
        "/api/grade",
        files={
            "image": ("sheet.jpg", io.BytesIO(sheet_bytes), "image/jpeg"),
            "answer_key_file": ("custom_key.csv", io.BytesIO(csv_bytes), "text/csv")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["questions"][0]["expected_answer"] == "D"
    assert data["questions"][1]["expected_answer"] == "E"
    assert data["questions"][2]["expected_answer"] == "A"


def test_nonexistent_result_export_404(client):
    """Test export endpoints return 404 for unknown result IDs."""
    res = client.get("/api/results/unknown-id-12345")
    assert res.status_code == 404

    csv_res = client.get("/api/export/unknown-id-12345.csv")
    assert csv_res.status_code == 404

    json_res = client.get("/api/export/unknown-id-12345.json")
    assert json_res.status_code == 404
