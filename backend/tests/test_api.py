"""Backend API tests."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_get_turbine_status(client):
    response = await client.get("/api/v1/turbine/status?turbine_id=SGT400-001")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["turbine_id"] == "SGT400-001"
    assert "health_score" in data["data"]
    assert "risk_level" in data["data"]


@pytest.mark.asyncio
async def test_get_anomaly_score(client):
    response = await client.get("/api/v1/predictions/anomaly?turbine_id=SGT400-001")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "result" in data
    assert "anomaly_score" in data["result"]


@pytest.mark.asyncio
async def test_get_failure_prediction(client):
    response = await client.get("/api/v1/predictions/failure?turbine_id=SGT400-001")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "failure_probability" in data["prediction"]
    assert "rul_days" in data["prediction"]
    assert "recommendation" in data["prediction"]


@pytest.mark.asyncio
async def test_get_alerts(client):
    response = await client.get("/api/v1/alerts/?turbine_id=SGT400-001")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "alerts" in data


@pytest.mark.asyncio
async def test_get_alert_summary(client):
    response = await client.get("/api/v1/alerts/summary?turbine_id=SGT400-001")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "total_active" in data["summary"]


@pytest.mark.asyncio
async def test_get_sensor_trend(client):
    response = await client.get(
        "/api/v1/turbine/trend/exhaust_gas_temp_c?turbine_id=SGT400-001&hours=24"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sensor"] == "exhaust_gas_temp_c"


@pytest.mark.asyncio
async def test_invalid_sensor(client):
    response = await client.get("/api/v1/turbine/trend/invalid_sensor")
    assert response.status_code == 400
