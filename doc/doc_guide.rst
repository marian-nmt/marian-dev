Writing documentation
---------------------

Marian’s documentation is generated using `Sphinx`_ + `Breathe`_ +
`Doxygen`_ + `Exhale`_. To put it simply, Marian uses `Doxygen`_ to
comment the source code and `Sphinx`_ (together with the extensions of
`Breathe`_ and `Exhale`_) to manage handwritten documentation and
library API reference.

If you are submitting new features, we would be very appreciative if you
could add some simple Doxygen comments (see the `Doxygen guidelines`_
below) along with your changes. Your doxygen comments will be integrated
in the documentation automatically.

There is an ongoing and incremental effort with the goal to document
essential Marian APIs in a consistent way. The existing code might not
follow these guidelines, but new code should.

Documentation with Doxygen
``````````````````````````

`Doxygen`_ is a powerful documentation system for C++ and many other
popular programming languages, such as C#, PHP, Java and Python. It
parses and extracts documentation from annotated C++ source code to
generate a comprehensive online documentation browser (in HTML) and/or
an offline reference manual (in LaTeX).

| **How to make the comments in your code known to doxygen?**
| Doxygen recognises several special comment blocks with some additional
  markings. In Marian, we follow the **Javadoc style**, which consist of
  a C-style comment block starting with two ``*``'s, like this:

.. code:: cpp

    /**
     * ... text ...
     */

For each entity in the code, there are basically two types of
descriptions, i.e., a *brief* description and *detailed* description. In
Marian, `JAVADOC_AUTOBRIEF`_ is set to YES in the configuration file,
therefore, using Javadoc style comment blocks will automatically start a
brief description which ends at the first dot followed by a space or new
line. Here is an example:

.. code:: cpp

    /** 
     *  Brief description which ends at this dot. Details follow
     *  here.
     */

If you want to put documentation after members (e.g., a variable and
enum), you have to put an additional ``<`` marker in the comment block.

.. code:: cpp

    int var; ///< Brief description after the member

To provide more details in the documentation, Doxygen supports a list of
special commands (keywords) which start with a backslash (\) or an
at-sign (@). In Marian, we recommend to use at-sign (@) to be consistent
with Javadoc style. See `Doxygen special commands`_ for all the
available commands. Here, we list the most common Doxygen commands we
use to document Marian:

+-----------------------+-----------------------+-----------------------+
| Doxygen Command       | Detailed Description  | Example               |
+=======================+=======================+=======================+
| @param                | Add a parameter       | ``@param device a po  |
|                       | description for a     | inter to the device`` |
|                       | function parameter    |                       |
+-----------------------+-----------------------+-----------------------+
| @return               | Add a return value    | ``@return a pointer t |
|                       | description for a     | o the constant node`` |
|                       | function              |                       |
+-----------------------+-----------------------+-----------------------+
| @see                  | Add a cross-reference | ``@see reshape()``    |
|                       | to classes,           |                       |
|                       | functions, methods,   |                       |
|                       | variables, files or   |                       |
|                       | URL.                  |                       |
+-----------------------+-----------------------+-----------------------+
| @ref                  | Create a reference to | ``@ref IndexType``    |
|                       | another item being    |                       |
|                       | documented.           |                       |
+-----------------------+-----------------------+-----------------------+
| @copybrief            | Copy the brief        | ``@copybrief slice``  |
|                       | description from the  |                       |
|                       | object specified      |                       |
+-----------------------+-----------------------+-----------------------+
| @copydetails          | Copy the detailed     | ``@copydetails dot``  |
|                       | documentation from    |                       |
|                       | the object specified  |                       |
+-----------------------+-----------------------+-----------------------+
| @note                 | Add a note message    | ``@note this is named |
|                       | where the text will   |  after an equivalent  |
|                       | be highlighted        | function in PyTorch`` |
+-----------------------+-----------------------+-----------------------+
| @warning              | Add a warning message | ``@warn               |
|                       | where the text will   | ing not implemented`` |
|                       | be highlighted        |                       |
+-----------------------+-----------------------+-----------------------+
| @b                    | Display a single word | ``@b bold``           |
|                       | using a bold font.    |                       |
+-----------------------+-----------------------+-----------------------+
| @c                    | Display a single word | ``@c void``           |
|                       | using a typewriter    |                       |
|                       | font.                 |                       |
+-----------------------+-----------------------+-----------------------+
| @p                    | Display a single word | ``@p transA``         |
|                       | using a typewriter    |                       |
|                       | font. Equivalent to   |                       |
|                       | ``@c``                |                       |
+-----------------------+-----------------------+-----------------------+
| @em                   | Display a single word | ``@em x``             |
|                       | in italics.           |                       |
+-----------------------+-----------------------+-----------------------+

.. note::

    Not all Doxygen special commands are supported in Exhale, e.g., `grouping`_. 
    Some commands like `@name`_ could lead to errors when parsing overloaded functions. 
    To free yourself from debugging the Doxygen comments for hours, we recommend you only using the above commands.

| **How to including math formulas in Doxygen?**
| Doxygen supports LaTeX math formulas (as images) in the documation. To
  include an inline formula that appears in the running text, we need
  wrap it by a pair of ``@f$`` commands. Here is an example:

.. code:: none

    Default is no smoothing, @f$\alpha = 0 @f$.  

This will result in: Default is no smoothing, |formula1|.

For the longer formulas which are in seperate lines, we can put ``\f[``
and ``\f]`` commands between the formulas. For instance:

.. code:: none

    @f[
       \operatorname{gelu}(x) = x \cdot \Phi(x)
         = x \cdot \frac{1}{2}\left[
            1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)
         \right]
         \sim \operatorname{swish}(x, 1.702)
    @f]

This will result in:

.. figure:: images/gelu_formula.png
   :alt: Example of formula 2

   Example of formula 2

.. note::

    Make sure the formula contains *valid* commands in `LaTeX’s math-mode`_.

| **What is a good pratice of Doxygen comments?**
| First of all, add Doxygen comments in the header files. You can find
  the examples of Doxygen comments in `src/graph/expression_graph.h`_.
  The basic rule of a good pratice is to keep Doxygen comments as
  intuitive and short as possible. Try not to introduce unnecessary
  vertical space (e.g., an empty line). A basic template of Doxygen
  comments is shown as follows:

.. code:: cpp

    /**
     * Brief summary.
     * Detailed description. More detail.
     * @see Some reference
     * @param <name> Parameter description.
     * @return Return value description.
     */

Documentation with Sphinx
`````````````````````````

Sphinx supports `Markdown`_ and `reStructuredText`_ documents. Our
handwritten documentations are located in `doc`_.
The default format of Sphinx is `reStructuredText`_ and 

most of the framework's power comes from the richness of its default
`reStructuredText`_
markup format.

Markdown
~~~~~~~~

reStructuredText
~~~~~~~~~~~~~~~~

.. _Sphinx: https://www.sphinx-doc.org/en/master/usage/quickstart.html
.. _Breathe: https://breathe.readthedocs.io/en/latest/directives.html
.. _Doxygen: http://www.doxygen.nl/manual/docblocks.html
.. _Exhale: https://exhale.readthedocs.io/en/latest/usage.html
.. _Doxygen guidelines: #documentation-with-doxygen
.. _JAVADOC_AUTOBRIEF: https://www.doxygen.nl/manual/config.html#cfg_javadoc_autobrief
.. _Doxygen special commands: https://www.doxygen.nl/manual/commands.html
.. _grouping: https://www.doxygen.nl/manual/grouping.html
.. _@name: https://www.doxygen.nl/manual/commands.html#cmdname
.. _LaTeX’s math-mode: https://en.wikibooks.org/wiki/LaTeX/Mathematics
.. _src/graph/expression_graph.h: https://github.com/marian-nmt/marian-dev/blob/master/src/graph/expression_graph.h
.. _Markdown: https://www.sphinx-doc.org/en/master/usage/markdown.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
.. _doc: https://github.com/marian-nmt/marian-dev/tree/master/doc
