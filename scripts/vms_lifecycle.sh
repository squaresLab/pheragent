#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

readonly PROGRAM_NAME="${0##*/}"
readonly PROJECT_TAG="mosip"
readonly ROLE_TAG_VALUES="nginx,core,rancher"
readonly EC2_REGION="us-east-1"

# Print command usage, required arguments, and examples.
usage() {
  cat <<EOF
Usage:
  ${PROGRAM_NAME} up --profile <SSO_PROFILE> [--auto-approve]
  ${PROGRAM_NAME} down --profile <SSO_PROFILE> [--auto-approve]
  ${PROGRAM_NAME} status --profile <SSO_PROFILE>
  ${PROGRAM_NAME} help

Commands:
  up       Start matching stopped EC2 instances, then show their status.
  down     Stop matching running EC2 instances, then show their status.
  status   Show matching EC2 instances in a table.
  help     Show this help message.

Every command targets nodes tagged project:${PROJECT_TAG} whose role tag is
nginx, core, or rancher.

Options:
  --profile <name>  Required AWS CLI SSO profile.
  --auto-approve    Skip the confirmation prompt for up or down.

Examples:
  ${PROGRAM_NAME} status --profile development
  ${PROGRAM_NAME} down --profile development
  ${PROGRAM_NAME} up --profile development --auto-approve

If the SSO session has expired, refresh it with:
  aws sso login --profile <name>
EOF
}

# Print an error and terminate with a non-zero exit status.
die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

# Verify that the AWS CLI is available before an AWS workflow starts.
require_aws() {
  command -v aws >/dev/null 2>&1 || die "AWS CLI v2 is required but was not found in PATH."
}

# Parse and validate options supplied after the command.
parse_arguments() {
  AWS_PROFILE=""
  AUTO_APPROVE=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        [[ $# -ge 2 && -n "$2" ]] || die "--profile requires a profile name."
        [[ -z "$AWS_PROFILE" ]] || die "--profile may only be specified once."
        AWS_PROFILE="$2"
        shift 2
        ;;
      --auto-approve)
        AUTO_APPROVE=true
        shift
        ;;
      *) die "Unknown argument: $1" ;;
    esac
  done

  [[ -n "$AWS_PROFILE" ]] || die "--profile <SSO_PROFILE> is required."
}

# Run every AWS command with the explicitly supplied SSO profile.
aws_cli() {
  aws --profile "$AWS_PROFILE" "$@"
}

# Validate the selected SSO profile and provide a login hint on failure.
check_aws_session() {
  if ! aws_cli sts get-caller-identity --output json >/dev/null 2>&1; then
    die "AWS authentication failed. Refresh the SSO session with: aws sso login --profile ${AWS_PROFILE}"
  fi
}

# Print all nodes matching the static role tags and mandatory project tag.
status_workflow() {
  local query

  query='Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,InstanceId:InstanceId,State:State.Name,Type:InstanceType,AvailabilityZone:Placement.AvailabilityZone,PrivateIp:PrivateIpAddress}'

  printf '\nEC2 nodes tagged project:%s and role:{nginx,core,rancher}\n' "$PROJECT_TAG"
  aws_cli ec2 describe-instances \
    --filters \
      "Name=tag:project,Values=${PROJECT_TAG}" \
      "Name=tag:role,Values=${ROLE_TAG_VALUES}" \
    --query "$query" \
    --output table
}

# Print only the instances selected for the pending start or stop action.
preview_affected_nodes() {
  local query

  query='Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,InstanceId:InstanceId,State:State.Name,Type:InstanceType,AvailabilityZone:Placement.AvailabilityZone,PrivateIp:PrivateIpAddress}'

  printf '\nAffected EC2 nodes\n'
  aws_cli ec2 describe-instances \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --query "$query" \
    --output table
}

# Find actionable instance IDs for the selected tags and lifecycle state.
load_action_instance_ids() {
  local state="$1"
  local instance_id

  INSTANCE_IDS=()
  while IFS= read -r instance_id; do
    [[ -n "$instance_id" && "$instance_id" != "None" ]] && INSTANCE_IDS+=("$instance_id")
  done < <(
    aws_cli ec2 describe-instances \
      --filters \
        "Name=tag:project,Values=${PROJECT_TAG}" \
        "Name=tag:role,Values=${ROLE_TAG_VALUES}" \
        "Name=instance-state-name,Values=${state}" \
      --query 'Reservations[].Instances[].InstanceId' \
      --output text | tr '\t' '\n'
  )
}

# Ask the operator to approve a mutation unless --auto-approve was supplied.
confirm_action() {
  local action="$1"
  local reply

  if [[ "$AUTO_APPROVE" == true ]]; then
    return
  fi

  printf '\nProceed with %s on %s instance(s)? [y/N] ' "$action" "${#INSTANCE_IDS[@]}"
  if ! read -r reply; then
    printf '\n'
    die "Confirmation input was not available."
  fi

  case "$reply" in
    y|Y|yes|YES|Yes) ;;
    *) printf 'Cancelled.\n'; exit 0 ;;
  esac
}

# Preview and start matching stopped nodes, then print their latest status.
up_workflow() {
  load_action_instance_ids stopped
  [[ ${#INSTANCE_IDS[@]} -gt 0 ]] || die "No stopped nodes match project:${PROJECT_TAG} and role:{nginx,core,rancher}."

  preview_affected_nodes
  confirm_action start
  printf '\nStarting %s instance(s)...\n' "${#INSTANCE_IDS[@]}"
  aws_cli ec2 start-instances \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'StartingInstances[].{InstanceId:InstanceId,Previous:PreviousState.Name,Current:CurrentState.Name}' \
    --output table
  status_workflow
}

# Preview and stop matching running nodes, then print their latest status.
down_workflow() {
  load_action_instance_ids running
  [[ ${#INSTANCE_IDS[@]} -gt 0 ]] || die "No running nodes match project:${PROJECT_TAG} and role:{nginx,core,rancher}."

  preview_affected_nodes
  confirm_action stop
  printf '\nStopping %s instance(s)...\n' "${#INSTANCE_IDS[@]}"
  aws_cli ec2 stop-instances \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'StoppingInstances[].{InstanceId:InstanceId,Previous:PreviousState.Name,Current:CurrentState.Name}' \
    --output table
  status_workflow
}

# Route the requested command to its workflow.
main() {
  local command="${1:-help}"

  case "$command" in
    help|-h|--help)
      usage
      ;;
    up|down|status)
      shift
      parse_arguments "$@"
      require_aws
      check_aws_session

      case "$command" in
        up) up_workflow ;;
        down) down_workflow ;;
        status) status_workflow ;;
      esac
      ;;
    *)
      printf 'Unknown command: %s\n\n' "$command" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"