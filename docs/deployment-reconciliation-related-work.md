# Related Work for Evidence-Grounded Deployment Reconciliation

This note compares systems that are directly relevant to HerAgentDeploy's
proposed loop:

```text
derive intent -> observe environment -> find gaps -> plan -> execute -> verify
```

It uses original papers, official documentation, specifications, and project
repositories. The comparison is about mechanisms HerAgent can reuse, not about
claiming that these projects solve the same research problem.

## System comparison

| System | Mechanism worth reusing | What it does not solve for HerAgent |
| --- | --- | --- |
| **Kubernetes controllers/operators** | Kubernetes resources separate desired `.spec` from observed `.status`; narrow controllers repeatedly observe and move one part of the system toward its desired state. HerAgent should similarly use small, repeatable reconcilers and re-observe after every action. [Kubernetes controllers](https://kubernetes.io/docs/concepts/architecture/controller/) | It assumes someone already authored the desired resources. It does not derive a system model from repositories and documentation or plan across heterogeneous environments. |
| **Terraform** | Configuration describes desired infrastructure, providers refresh real objects, state binds declarations to remote identities, and `plan` computes reviewable create/update/delete actions before `apply`. HerAgent should retain identity bindings and separate inspect, plan, approve, and apply. [Terraform workflow](https://developer.hashicorp.com/terraform/cli/run), [state](https://developer.hashicorp.com/terraform/language/state) | It manages declared provider resources. It does not discover undeclared application components, infer application health, or repair arbitrary deployment scripts. |
| **Crossplane** | A composite resource expresses a high-level capability while a Composition resolves it into environment-specific resources. Composition functions receive observed and desired resources, accumulate desired state, and can request extra evidence in a bounded loop. HerAgent can express intent as capabilities and resolve providers after inspecting the environment. [Crossplane Compositions](https://docs.crossplane.io/master/concepts/compositions/), [Composition revisions](https://docs.crossplane.io/latest/composition/composition-revisions/) | Schemas, providers, and Compositions must already exist. Crossplane does not mine an unfamiliar repository or determine whether an undocumented installer is correct. |
| **Open Application Model and KubeVela** | OAM models an application using components, workload types, operational traits, scopes, and configuration. KubeVela adds policies, health checks, workflows, dependencies, and approval steps. This is a useful compact shape for separating application intent from environment-specific operations. [OAM specification](https://github.com/oam-dev/spec), [OAM terminology](https://github.com/oam-dev/spec/blob/master/2.overview_and_terminology.md), [KubeVela application model](https://kubevela.io/docs/getting-started/core-concept/) | It does not automatically discover components or recover an unknown deployment procedure. Its implementation is primarily Kubernetes-centered. |
| **Ansible** | Playbooks combine declared final states with ordered multi-host tasks. Most modules first test current state and avoid changes when the goal is already met; check and diff modes preview supported actions. HerAgent should expose executor adapters as `observe`, `check`, and `apply`, and prefer idempotent operations. [Ansible playbooks and idempotency](https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_intro.html), [check and diff modes](https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_checkmode.html) | Not every module or playbook is idempotent, and check mode cannot reliably simulate arbitrary shell commands. Ansible does not continuously construct or revise a deployment model. |
| **NixOS and GNU Guix** | Declarative inputs produce versioned system generations. New configurations can be tested, promoted, or rolled back without inventing reverse shell commands. HerAgent should version its working source, plan, and accepted deployment artifact, and promote only validated revisions. [NixOS manual](https://nixos.org/manual/nixos/stable/), [How Nix works](https://nixos.org/guides/how-nix-works/), [Guix manual](https://guix.gnu.org/manual/en/guix.pdf) | Their guarantees apply mainly to content managed by their functional stores. They do not roll back arbitrary external or stateful distributed services and do not discover undeclared deployments. |
| **MAPE-K/autonomic computing** | MAPE-K separates Monitor, Analyze, Plan, and Execute around shared Knowledge. This gives HerAgent a clear boundary: harness tools monitor and execute; the model analyzes and proposes plans; an evidence store carries knowledge between iterations. [IBM autonomic-computing blueprint](https://www.jroller.org/autonomic/pdfs/ACwpFinal.pdf), [Kephart and Chess](https://doi.org/10.1109/MC.2003.1160055) | It is an architectural reference model, not a deployment implementation. It does not define evidence precedence, state schemas, LLM safety, or approval rules. |
| **Flux/GitOps** | Flux continuously reconciles versioned Git or OCI declarations with live Kubernetes resources, reports health separately from synchronization, tracks revisions, supports dependencies, and allows selected fields to be ignored or owned by other controllers. HerAgent should compare only fields it owns and keep plan revisions auditable. [Flux concepts](https://fluxcd.io/flux/concepts/), [Kustomization reconciliation](https://fluxcd.io/flux/components/kustomize/kustomizations/) | It assumes the desired manifests already exist. It does not derive them from source material or safely revise intent when execution reveals a missing prerequisite. |
| **K8sGPT** | Deterministic Kubernetes analyzers first extract compact findings; an LLM then explains them. This supports HerAgent's use of authoritative probes and focused evidence instead of raw cluster dumps. [K8sGPT repository](https://github.com/k8sgpt-ai/k8sgpt) | It primarily diagnoses Kubernetes resources. It does not construct desired state, reconcile deployments across environments, or produce a reproducible end-to-end deployment. |
| **HolmesGPT** | The SRE agent iteratively uses read-only toolsets across Kubernetes, clouds, Helm, databases, logs, and metrics. Its least-privilege investigation model supports a specialized HerAgent debugging role. [HolmesGPT repository](https://github.com/HolmesGPT/holmesgpt), [read-only tool design](https://github.com/HolmesGPT/holmesgpt/blob/master/docs/why-holmesgpt.md) | Its central task is incident investigation, not authoritative desired-state construction or controlled deployment convergence. |

## Design consequences for HerAgent

### Desired state

Use a short capability-and-outcome declaration, borrowing from Crossplane and
OAM. A human supplies only goals, constraints, and policies. Source analysis
may propose components and acceptance checks, but these remain evidence-backed
proposals. The desired state should not repeat complete Helm values or
Kubernetes manifests.

### Observed state and equivalence

Represent each relevant resource or capability with a stable identity, selected
properties, health, evidence source, and observation time. Equivalence should
be a set of per-goal predicates, not equality between two large documents:

```text
structurally present AND configured as required AND healthy
```

Ignore volatile and unowned fields, as Flux and Kubernetes controllers do.
Report `satisfied`, `unsatisfied`, or `unknown`; a failed probe yields
`unknown`, never evidence of absence.

### Reconciliation evidence

"Deterministic evidence wins" must be scoped to the fact's domain:

- A successful runtime probe is authoritative about what currently exists.
- A structured source artifact is authoritative about what that artifact
  declares.
- A human goal is authoritative about what should exist.
- Documentation is evidence of intended procedure, not proof of current state.
- An LLM conclusion is a proposal, not evidence.

This prevents a runtime observation from erasing a human goal while allowing it
to correct a stale claim such as "persistent storage is already provided."

### Agent and harness boundary

The harness owns tools, permissions, evidence capture, deterministic checks,
state transitions, approval, execution, and bounded retries. The model receives
compact evidence and proposes queries, model updates, plans, diagnoses, or
patches. Only the harness can accept and apply those proposals.

## Research gap

Within the reviewed primary sources, no single system combines all of the
following:

1. deriving an initial deployment model from repositories and documentation;
2. observing heterogeneous, partially declared environments;
3. reconciling conflicts using provenance-aware evidence;
4. revising the model and plan after runtime failures; and
5. producing a reviewed, reproducible deployment artifact.

That integration, rather than the control-loop concept alone, is the defensible
research space for HerAgentDeploy.
