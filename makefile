# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help install test docker tree check demographics brief personas

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

##@ General

help:  ## Show the available targets and their arguments
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@ / {printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} \
		/^[0-9a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' makefile
	@printf "\n\033[1mPersona commands take flags, so run them directly\033[0m\n"
	@printf "  \033[36m%-34s\033[0m %s\n" \
		"uv run personas demographics --rows 1000" "Build a dataset of that size" \
		"uv run personas brief --rows 5" "Print five plain records" \
		"uv run personas generate --rows 2 --live" "Write LLM personas for two records" \
		"uv run personas --help" "Every command and flag"

install: ## Install dependencies
	@echo "Installing the 'danish_personas' project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@$(MAKE) --quiet add-repo-to-git
	@echo "Installed the 'danish_personas' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a 'uv' project. Use 'uv add <package>' to install new dependencies and 'uv remove <package>' to remove them."

install-non-interactive:
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables-non-interactive
	@$(MAKE) --quiet setup-git
	@$(MAKE) --quiet add-repo-to-git

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

install-dependencies:
	@uv python install 3.14
	@uv sync --all-extras --all-groups --python 3.14

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name "${GIT_NAME}"
	@git config --local user.email "${GIT_EMAIL}"

add-repo-to-git:
	@if [ ! "$(shell git status --short)" = "" ] && [ "$(shell git --no-pager log --all | sed 's/`//g')" = "" ]; then \
		git add .; \
		git commit --quiet -m "Initial commit"; \
	fi
	@if [ "$(shell git remote)" = "" ]; then \
		git remote add origin git@github.com:syv-ai/danish_personas.git; \
	fi

##@ Development

test:  ## Run tests
	@uv run pytest && \
		uv run readme-cov && \
		git add README.md && \
		git commit -m "docs: Update coverage badge"

docker:  ## Build Docker image and run container
	@make install
	@docker build -t danish_personas .
	@docker run -it --rm danish_personas

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

check:  ## Lint, format, and type-check the code
	@git add . && uv run pre-commit run --all-files; status=$$?; git reset >/dev/null; exit $$status

##@ Personas

# These are shortcuts for the defaults only. Anything with arguments goes to the CLI
# directly, which takes ordinary flags: uv run personas <command> --help
demographics:  ## Build the statistical dataset
	@uv run personas demographics

brief:  ## Print plain persona records from the dataset
	@uv run personas brief

personas:  ## Plan LLM personas for the dataset
	@uv run personas generate
