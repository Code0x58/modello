workflow "Publish on push to master" {
  on = "push"
  resolves = ["Release"]
}

action "Test" {
  uses = "docker://python:3.7-slim"
  args = "python setup.py test"
}

action "Filter release" {
  needs = ["Test"]
  uses = "actions/bin/filter@95c1a3b"
  args = "tag v*"
}

action "Release" {
  uses = "docker://code0x58/action-python-publish:master"
  needs = ["Filter release"]
  secrets = ["TWINE_PASSWORD", "TWINE_USERNAME"]
}
