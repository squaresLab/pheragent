from __future__ import annotations

from pathlib import Path

from ..enums import InventoryCategory
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import DeterministicParser, ParseResult
from .config import ConfigurationParser
from .markdown import MarkdownParser
from .shell import ShellParser
from .terraform import TerraformParser
from .yaml_files import YamlDeploymentParser

_YAML_CATEGORIES = {
    InventoryCategory.ANSIBLE,
    InventoryCategory.CI_WORKFLOW,
    InventoryCategory.COMPOSE,
    InventoryCategory.HELM,
    InventoryCategory.HELMSMAN,
    InventoryCategory.KUBERNETES,
    InventoryCategory.KUSTOMIZE,
}


class DeterministicParserRegistry:
    def __init__(self) -> None:
        self.markdown = MarkdownParser()
        self.shell = ShellParser()
        self.terraform = TerraformParser()
        self.yaml = YamlDeploymentParser()
        self.configuration = ConfigurationParser()

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        parser = self._parser_for(entry)
        return parser.parse(source, entry, evidence)

    def _parser_for(self, entry: InventoryEntry) -> DeterministicParser:
        if entry.category == InventoryCategory.DOCUMENTATION:
            return self.markdown
        if entry.category == InventoryCategory.SHELL:
            return self.shell
        if entry.category == InventoryCategory.TERRAFORM:
            return self.terraform
        if entry.category in _YAML_CATEGORIES:
            if Path(entry.path).suffix.lower() in {".yaml", ".yml"}:
                return self.yaml
            return self.configuration
        if entry.category == InventoryCategory.CONFIGURATION:
            if Path(entry.path).suffix.lower() in {".yaml", ".yml"}:
                return self.yaml
            return self.configuration
        raise ValueError(f"no deterministic parser for {entry.category}: {entry.path}")
