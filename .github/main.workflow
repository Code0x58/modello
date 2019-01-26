workflow "Test and Publish" {
  on = "push"
  resolves = ["Release"]
}

action "Test" {
  uses = "./.github/actions/test/"
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
