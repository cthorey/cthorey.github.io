# Using bretfisher jekyll image to build/serve
MAKEFLAGS += --warn-undefined-variables --no-print-directory
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:

REPO = cthorey.github.io
GIT_COMMIT := $(shell git rev-parse HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_SHORT_COMMIT := $(shell git rev-parse --short HEAD)

.PHONY: help
help:
	$(info Available make targets:)
	@egrep '^(.+)\:\ ##\ (.+)' ${MAKEFILE_LIST} | column -t -c 2 -s ':#'

.PHONY: serve
serve: ## serve website locally
	$(info *** serve website)
	@docker run \
		-v $(PWD):/site \
    -p 8080:4000 \
		bretfisher/jekyll-serve

.PHONY: deploy
deploy: ## deploy to github pages
	$(info *** deploy to github pages)
	./bin/deploy

.PHONY: build
build: ## build website locally
	$(info *** build website)
	@docker run \
		-v $(PWD):/site \
    -p 8080:4000 \
		bretfisher/jekyll-serve bundle exec jekyll build
