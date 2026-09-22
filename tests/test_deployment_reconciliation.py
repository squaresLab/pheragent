from pheragent.deployment.analysis_models import DeploymentContext
from pheragent.deployment.reconciliation import (
    CapabilityStatus,
    apply_runtime_reconciliation,
    reconcile_provided_capabilities,
)
from pheragent.deployment.runtime_context import RuntimeContextSnapshot, RuntimeProbe


def test_reconciliation_uses_generic_runtime_evidence() -> None:
    capabilities = [
        "linux-host",
        "persistent-storage",
        "csi-provisioning",
        "helm-release:istio-system/istio",
        "helm-release:missing",
        "ingress",
    ]
    context = DeploymentContext.model_validate(
        {
            "system": "fixture",
            "provided_blocks": [
                {
                    "id": "B0",
                    "type": "runtime_environment",
                    "subtype": "platform_services",
                    "provides": capabilities,
                }
            ],
        }
    )
    snapshot = RuntimeContextSnapshot(
        captured_at="2026-09-22T10:00:00+00:00",
        host={"available": True, "system": "Linux"},
        aws={"available": False},
        kubernetes={
            "available": True,
            "storage_classes": [
                {
                    "kind": "StorageClass",
                    "name": "default",
                    "state": "driver.example.io",
                    "is_default": True,
                }
            ],
            "csi_drivers": [{"kind": "CSIDriver", "name": "driver.example.io"}],
            "helm_releases": [
                {
                    "kind": "HelmRelease",
                    "name": "istio",
                    "namespace": "istio-system",
                    "state": "deployed",
                }
            ],
        },
        probes=[
            RuntimeProbe(
                provider=provider,
                name=name,
                command=[],
                succeeded=True,
                duration_seconds=0,
            )
            for provider, name in (
                ("host", "system"),
                ("kubernetes", "storage_classes"),
                ("kubernetes", "csi_drivers"),
                ("kubernetes", "helm_releases"),
            )
        ],
    )

    reconciliation = reconcile_provided_capabilities(context, snapshot)
    statuses = {check.capability: check.status for check in reconciliation.checks}

    assert statuses == {
        "linux-host": CapabilityStatus.SATISFIED,
        "persistent-storage": CapabilityStatus.SATISFIED,
        "csi-provisioning": CapabilityStatus.SATISFIED,
        "helm-release:istio-system/istio": CapabilityStatus.SATISFIED,
        "helm-release:missing": CapabilityStatus.MISSING,
        "ingress": CapabilityStatus.UNKNOWN,
    }
    effective = apply_runtime_reconciliation(context, reconciliation)
    assert effective.provided_blocks[0].provides == [
        capability for capability in capabilities if capability != "helm-release:missing"
    ]
