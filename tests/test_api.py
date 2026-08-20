from pathlib import Path

from fastapi.testclient import TestClient

from enterprise_rag.api.main import create_app


def test_health_endpoint_on_empty_project(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "app.yaml").write_text("app: {}\ndatabase: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "embeddings.yaml").write_text("embedding: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "retrieval.yaml").write_text("chunking: {}\nretrieval: {}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json()["status"] == "ok"


def test_web_root_is_served_from_project():
    project_root = Path(__file__).resolve().parents[1]
    client = TestClient(create_app(project_root))
    response = client.get("/")
    assert response.status_code == 200
    assert "Northstar" in response.text
    assert "Add documents" in response.text
    assert "Attach" in response.text
    assert "Voice" in response.text


def test_document_upload_refreshes_runtime_index(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "app.yaml").write_text("app: {}\ndatabase: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "embeddings.yaml").write_text("embedding: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "retrieval.yaml").write_text("chunking: {}\nretrieval: {}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/documents/upload",
        files={"file": ("travel.md", "---\ndocument_id: uploaded_travel\ntitle: Uploaded Travel\ndepartment: Finance\n---\n\n# Limits\nHotel allowance is EUR 200 in Lyon.\n", "text/markdown")},
    )
    assert response.status_code == 200
    assert response.json()["documents_added"] == 1
    assert client.get("/health").json()["documents"] == 1
    assert client.get("/documents").json()[0]["document_id"] == "uploaded_travel"
    image_response = client.post(
        "/documents/upload",
        files={"file": ("diagram.png", b"fake-image-bytes", "image/png")},
    )
    assert image_response.status_code == 200
    assert image_response.json()["kind"] == "image"
    assert image_response.json()["indexed"] is False


def test_api_key_protects_private_endpoints(tmp_path: Path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "app.yaml").write_text("app: {}\ndatabase: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "embeddings.yaml").write_text("embedding: {}\n", encoding="utf-8")
    (tmp_path / "configs" / "retrieval.yaml").write_text("chunking: {}\nretrieval: {}\n", encoding="utf-8")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "test-secret")
    client = TestClient(create_app(tmp_path))
    assert client.get("/health/live").status_code == 200
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "test-secret"}).status_code == 200
