from __future__ import annotations

import json
from pathlib import Path

import yaml

from pheragent.deployment.enums import DeterministicFindingKind
from pheragent.deployment.inspection import run_deterministic_inspection


def _write_fixture(source: Path) -> None:
    (source / ".github" / "workflows").mkdir(parents=True)
    (source / "playbooks").mkdir()
    (source / "README.md").write_text(
        "# Install\n\n1. Prepare hosts\n\n```bash\nkubectl get nodes\n```\n",
        encoding="utf-8",
    )
    (source / "install.sh").write_text(
        "#!/bin/sh\n"
        "helm upgrade --install demo ./chart\n"
        "kubectl wait --for=condition=Ready pod/demo\n",
        encoding="utf-8",
    )
    (source / "compose.yaml").write_text(
        "services:\n"
        "  database:\n"
        "    image: postgres:16\n"
        "  api:\n"
        "    image: demo/api\n"
        "    depends_on:\n"
        "      - database\n",
        encoding="utf-8",
    )
    (source / ".github" / "workflows" / "deploy.yml").write_text(
        "name: Deploy\n"
        "on: workflow_dispatch\n"
        "jobs:\n"
        "  prepare:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo prepare\n"
        "  deploy:\n"
        "    needs: prepare\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Deploy chart\n"
        "        run: helm upgrade --install demo ./chart\n",
        encoding="utf-8",
    )
    (source / "deployment.yaml").write_text(
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: demo\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "        - name: demo\n"
        "          image: demo/api\n"
        "          readinessProbe:\n"
        "            httpGet:\n"
        "              path: /health\n"
        "              port: 8080\n"
        "      volumes:\n"
        "        - name: application-data\n"
        "          emptyDir: {}\n",
        encoding="utf-8",
    )
    (source / "main.tf").write_text(
        'resource "aws_instance" "node" {\n'
        "  depends_on = [module.network]\n"
        "}\n"
        'output "node_ip" {\n'
        "  value = aws_instance.node.public_ip\n"
        "}\n",
        encoding="utf-8",
    )
    (source / "Chart.yaml").write_text(
        "apiVersion: v2\n"
        "name: demo-chart\n"
        "version: 1.0.0\n"
        "dependencies:\n"
        "  - name: postgresql\n"
        "    version: 16.0.0\n"
        "    repository: https://charts.example.test\n",
        encoding="utf-8",
    )
    (source / "kustomization.yaml").write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        "resources:\n"
        "  - deployment.yaml\n",
        encoding="utf-8",
    )
    (source / "deployment.dsf.yaml").write_text(
        "namespaces:\n  demo: {}\napps:\n  demo-api:\n    chart: demo/chart\n    namespace: demo\n",
        encoding="utf-8",
    )
    (source / "playbooks" / "site.yml").write_text(
        "- name: Prepare hosts\n  hosts: cluster\n  roles:\n    - container-runtime\n",
        encoding="utf-8",
    )


def test_deterministic_inspection_extracts_cross_format_findings(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_fixture(source)
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "sources": [
                    {
                        "id": "fixture",
                        "kind": "local_directory",
                        "location": "source",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = run_deterministic_inspection(
        sources_path=sources_path,
        output_dir=output,
        strict=True,
    )
    findings = result.findings

    assert any(
        finding.kind == DeterministicFindingKind.DOCUMENT_HEADING and finding.name == "Install"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.DEPENDENCY
        and finding.attributes.get("source") == "api"
        and finding.attributes.get("target") == "database"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.DEPENDENCY
        and finding.attributes.get("source") == "deploy"
        and finding.attributes.get("target") == "prepare"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.RESOURCE
        and finding.attributes.get("kubernetes_kind") == "Deployment"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.VALIDATION
        and finding.attributes.get("probe") == "readinessProbe"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.RESOURCE
        and finding.name == "application-data"
        and finding.attributes.get("kubernetes_field") == "volumes"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.RESOURCE
        and finding.name == "resource.aws_instance.node"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.COMPONENT and finding.name == "demo-chart"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.DEPENDENCY
        and finding.name == "deployment.yaml"
        and finding.attributes.get("kustomize_field") == "resources"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.COMPONENT and finding.name == "demo-api"
        for finding in findings
    )
    assert any(
        finding.kind == DeterministicFindingKind.PROCEDURE_STEP and finding.name == "Prepare hosts"
        for finding in findings
    )
    assert result.evidence_count > 0
    evidence_records = {
        record["id"]: record
        for line in (output / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
        for record in [json.loads(line)]
    }
    assert all(
        reference in evidence_records for finding in findings for reference in finding.evidence_refs
    )
    workflow_dependency = next(
        finding
        for finding in findings
        if finding.attributes.get("source") == "deploy"
        and finding.attributes.get("target") == "prepare"
    )
    workflow_evidence = evidence_records[workflow_dependency.evidence_refs[0]]
    assert workflow_evidence["start_line"] == 9
    assert workflow_evidence["end_line"] == 9
    assert workflow_evidence["excerpt"].strip() == "needs: prepare"
    assert all(entry.inspected for entry in result.inventory.entries if entry.selected)

    repeated_output = tmp_path / "repeated-output"
    run_deterministic_inspection(
        sources_path=sources_path,
        output_dir=repeated_output,
        strict=True,
    )
    for filename in (
        "source-manifest.json",
        "repository-inventory.json",
        "evidence.jsonl",
        "deterministic-findings.jsonl",
    ):
        assert (repeated_output / filename).read_bytes() == (output / filename).read_bytes()
