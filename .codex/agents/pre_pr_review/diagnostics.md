# Results, errors, and diagnostic inspection

Use this reference when a change affects validation results, exceptions,
provenance, introspection, or their nested displays.

- Keep result states and guarantees precise. Unknown validity, assumed history,
  unchecked inputs, and checks that pass or fail may convey different
  information. Verify the actual contract before using truth negation or
  treating no error as proof. Names, properties, and consumers must preserve
  those distinctions.
- Check that meaningful library error categories can be distinguished from one
  another and from upstream failures. Reuse the established hierarchy and expose
  relevant values as structured attributes for callers to inspect or match,
  rather than requiring exception-message parsing. Add distinctions only when
  they serve a real caller need.
- Preserve the origin of observations through nested construction and
  inspection. A value inferred within a child is different from a value
  supplied to that child; a real child input can remain an input when the child
  is supplied to a parent. Check the scope in which each label is claimed.
- Make nested diagnostic displays readable without losing identity. Trim
  redundant local path prefixes where useful, while keeping cross-component
  references unambiguous and supplied/inferred labels faithful to their data.
- Keep expensive diagnostic analysis an explicit caller choice, usable in a
  REPL or exception path. Check that routine construction or exception handling
  does not silently acquire that analysis cost. Ordinary cheap validation can
  still run as part of the supported operation.

Check representative consumers of the result/error contract as well as its
producer. Use the Sphinx reference for teaching the returned data and the test
reference for coverage of its states; keep those concerns in their own review
areas.
