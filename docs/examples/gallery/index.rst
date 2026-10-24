Explore the fundamentals
========================

The gallery is the interactive companion to :doc:`../../getting-started`.
It follows the same four concepts and uses the same kinds of questions.

.. code-block:: console

   $ pip install 'modello[gui]'
   $ python -m examples.gallery

From a checkout, use ``uv run --extra gui python -m examples.gallery``.
Open http://127.0.0.1:5070. Choose a model, inspect **Equations**, and select
**Explore a question**. The navigation leads to the next concept.

.. list-table:: Four fundamentals
   :header-rows: 1
   :widths: 25 40 35

   * - Concept
     - Question to express
     - Starting result
   * - Scalar models
     - Size a brace or infer a missing side.
     - A 3 cm run and 4 cm rise give a 5 cm brace.
   * - Component models
     - Determine a quantity inside a parcel's base from its full clearance.
     - A 3 × 4 cm base and 12 cm height give 13 cm clearance.
   * - Vector models
     - Evaluate or fit a line against sensor readings.
     - The proposed line has squared error 1; fitting reduces it to 0.7.
   * - Recurrent models
     - Post fills or infer a sale price from a closing balance.
     - The account closes with 4 shares and USD 908.

The same five questions
-----------------------

Every model starts with a numeric result and offers:

1. **Calculate results**: follow the known inputs to their consequences.
2. **Infer missing inputs**: release inputs and supply an outcome or fitting
   condition to determine them.
3. **Try different inputs**: change measurements or collection length while
   keeping the same equations.
4. **Leave inputs unknown**: retain a formula in the missing quantities.
5. **Check the observations**: try a deliberately conflicting claim.

Each question starts from the original observations. Questions do not accumulate
constraints from earlier questions. An inverse question releases the quantity
it is inferring.

A rejected question keeps its explanation visible and preserves the last
successful results. Choose **Calculate results** to return to the starting case.
Use **Reset inputs** to discard custom changes.

Apply the ideas to your own problem
-----------------------------------

The :doc:`Python walkthrough <../../getting-started>` shows the declarations
behind each concept. Use :doc:`components <../../nesting>` to reuse a model,
:doc:`vectors <../../vectors>` for measurements and objectives, and
:doc:`model vectors <../../model-vectors>` for related states.

The separate **Extensions** page contains the opt-in
:doc:`Itô example <../../contrib/stochastic>`. The :doc:`GUI guide <../../gui>`
shows how to open your own model in the same explorer.
