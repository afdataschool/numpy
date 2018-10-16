---
title: "Formatting"
teaching: 10
exercises: 0
questions:
- "How are Software and Data Carpentry lessons formatted?"
objectives:
- "Explain the header of each episode."
- "Explain the overall structure of each episode."
- "Explain why blockquotes are used to format parts of episodes."
- "Explain the use of code blocks in episodes."
keypoints:
- "Lesson episodes are stored in _episodes/dd-subject.md."
- "Each episode's title must include a title, time estimates, motivating questions, lesson objectives, and key points."
- "Episodes should not use sub-headings or HTML layout."
- "Code blocks can have the source, regular output, or error class."
- "Special sections are formatted as blockquotes that open with a level-2 header and close with a class identifier."
- "Special sections may be callouts or challenges; other styles are used by the template itself."
---
---
layout: page
title: Advanced NumPy 
subtitle: Array container
minutes: 30
---
> ## Learning Objectives {.objectives}
>
> After the lesson learner:
>
> * Can list some of the object types which can be contained in an array.
> * Understands the concept of `dtype` and can select `dtype` best for the data at hand.
> * Knows what is an object array and when it is created.
> * Can explain what are `ndim`, `shape` and `stride` properties of an array.
> * Understand the layout of an array in memory and knows how to use it for best array performance.
> * Can explain the difference between Fortran- and C-based order. Knows the default.

### Data type 

In contrast to built-in Python containers (like lists)  NumPy arrays can store elements of pre-determined type only. To see the type of array contents you can use the `dtype` attribute. Let's look at two examples:

```
>>> a = np.array([1, 2, 3])
>>> a.dtype
dtype('int64')

>>> b = np.array([1., 2., 3.])
>>> b.dtype
dtype('float64')
```

In the first case the numbers are 64-bit (8-byte) integers and in the second 64-bit floating point (real)  numbers. Note that NumPy auto-detects the data-type from the input. Specialised data types allow us to store data more compactly in memory, but most of the time we simply work with floating point numbers.

Note that all of the elements of an array must be of the same type. If we construct an array with different elements it will be **cast** to the "most general" type that can represent all elements. For example, array constructed from real numbers and integers will have a floating point data type:

```
>>> a = np.array([1., 2])
dtype('float64')
```

In case it is impossible, NumPy will use an `object` type (also represented by capital `'O'`) which can represent any Python object -- even a function:

```
>>> def f(): pass
>>> a = np.array([f, f])
>>> a.dtype
dtype('O')
```

Some of NumPy features (like element-wise functions, `np.abs`, `np.sqrt`, etc., or reductions, `np.sum`, `np.max` etc.) won't work with object arrays, but all types of indexing still work.

`object` type is most commonly encountered when constructing an array from multiple lists of different lengths:

```
>>> np.array([[1], [2, 3]])
array([[1], [2, 3]], dtype=object)
```


> ## Integer or real number? {.challenge}
>
> Construct the array `x = np.array([0, 1, 2, 255], dtype=np.uint8)` (here, `uint8`
> represents a single byte in memory, an unsigned integer between 0 and 255). Can
> you explain the results obtained by x + 1 and x / 2? Also try `x.astype(float) + 1` and `x.astype(float) / 2`.

> ## Data types {.challenge}
>
> Try to guess the data type of the following arrays. Then test your prediction by  constructing the arrays and check their dtype attribute.
>
> ```
> a = np.array([[1, 2], 
>               [2, 3]])
> b = np.array(['a', 'b', 'c'])
> c = np.array([1, 2, 'a'])
> d = np.array([np.dot, np.array])
> e = np.random.randn(5) > 0
> f = np.arange(5)
> ```

> ## Complex data types {.challenge}
>
> Imagine you have a list of names and scores, which you want to store in numpy array. Construct a dtype such that the following works. Look at documentation of `np.dtype`.
>
> ```
> dtype = ?
> np.array([('Bartosz', 5), ('Nelle', 4)], dtype=dtype)
> ```

### Memory layout

NumPy array is just a memory block with extra information how to interpret its contents. Since memory has only linear address space, NumPy arrays need extra information how to lay out this block into multiple dimensions. This is done by means of `shape`  and `strides` attributes:

![Shape and strides](fig/strides.svg)

Lets try to reproduce this example. We first generate a 1D NumPy array of 8 elements:

```
>>> a = np.arange(8, dtype=np.uint8)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint8)
>>> a.strides
(1,)
>>> a.shape
(8,)
```

`shape` and `strides` attributes are read-only, so we can not modify them directly. However, we my use 
`as_strided` function from NumPy library module:

```
>>> a1 = np.lib.stride_tricks.as_strided(a, strides=(4, 1), shape=(2,4))
>>> a1
array([[0, 1, 2, 3],
       [4, 5, 6, 7]], dtype=uint8)
```

Similarly, we can obtain the second example:
```
>>> a2 = np.lib.stride_tricks.as_strided(a, strides=(2, 1), shape=(3,4))
>>> a2
array([[0, 1, 2, 3],
       [2, 3, 4, 5],
       [4, 5, 6, 7]], dtype=uint8)
```

Note that in the second case the same data appears twice. However, it does not consume extra memory -- all three arrays share the same memory block:

```
>>> a[2] = 100
>>> a1
array([[  0,   1, 100,   3],
       [  4,   5,   6,   7]], dtype=uint8)
>>> a2
array([[  0,   1, 100,   3],
       [100,   3,   4,   5],
       [  4,   5,   6,   7]], dtype=uint8)
```

> ## Transpose {.challenge}
>
> Create 3x4 random array. Have a look at its different properties: ``x.shape``, ``x.ndim``,
``x.dtype``, ``x.strides``. What does each property tell you about the array?
> 
> Compare the strides property of x.T to the above. What is x.T and can you
explain the new strides?

> ## Fastest changing index {.challenge}
>
>  Compare the time of summing over rows and columns of an array `A = np.random.rand(10, 100000)`. How would you explain the differences? (*Hint*: To measure evaluation time you can use `%timeit` of ipython)

> ## Sliding window {.challenge}
>
> Use `as_strided` to produce a sliding-window view of a 1D array.
>
> ```
> def sliding_window(arr, size=2):
>     """Produce an array of sliding window views of `arr`
>     
>     Parameters
>     ----------
>     arr : 1D array, shape (N,)
>         The input array.
>     size : int, optional
>         The size of the sliding window.
>         
>     Returns
>     -------
>     arr_slide : 2D array, shape (N - size - 1, size)
>         The sliding windows of size `size` of `arr`.
>         
>     Examples
>     --------
>     >>> a = np.array([0, 1, 2, 3])
>     >>> sliding_window(a, 2)
>     array([[0, 1],
>            [1, 2],
>            [2, 3]])
>     """
>     return arr  # fix this
> ```

> ## Fortran or C-ordering? {.challenge}
>
> The `order` keyword of some `numpy` functions determines how two- or more dimensional arrays are laid out in the memory. If order is 'C', then the array will be in C-contiguous order (last-index varies the fastest). If order is 'F', then the returned array will be in Fortran-contiguous order (first-index varies the fastest). In what order will be the 2D array stored in memory? (*Hint*: You can use `np.ravel(x, order='A')` to test it).

> ## Broadcasting revisited {.challenge}
>
>  Explain how broadcasting works internally using the example below. What will be shapes and strides of `x` and `y` after broadcasting. Test it using `np.broadcast_arrays` in the following example and look at `strides` and `shape` properties of both arrays.
>
> ```
> x = np.random.rand(5, 10)
> y = np.random.rand(10)
> z = x + y
>
> xb, yb = np.broadcast_arrays(x, y)
> ```
A lesson consists of one or more episodes,
each of which has:

*   a [YAML][yaml] header containing required values
*   some teachable content
*   some exercises

The diagram below shows the internal structure of a single episode file
(click on the image to see a larger version):

<a href="{{ page.root }}/fig/episode-format.png">
  <img src="{{ page.root }}/fig/episode-format-small.png" alt="Formatting Rules" />
</a>

## Maximum Line Length

Limit all lines to a maximum of 100 characters.
`bin/lesson_check.py` will report lines longer than 100 characters
and this can block your contributions of being accepted.

The two reasons behind the decision to enforce a maximum line lenght are
(1) make diff and merge easier in the command line and other user interfaces
and
(2) make update of translation of the lessons easier.

## Locations and Names

Episode files are stored in `_episodes`
or, for the case of R Markdown files, `_episodes_rmd`
so that [Jekyll][jekyll] will create a [collection][jekyll-collection] for them.
Episodes are named `dd-subject.md`,
where `dd` is a two-digit sequence number (with a leading 0)
and `subject` is a one- or two-word identifier.
For example,
the first three episodes of this example lesson are
`_episodes/01-design.md`,
`_episodes/02-tooling.md`
and `_episodes/03-formatting.md`.
These become `/01-design/index.html`, `/02-tooling/index.html`, and `/03-formatting/index.html`
in the published site.
When referring to other episodes, use:

{% raw %}
    [link text]({{ page.root }}{% link _episodes/dd-subject.md %})
{% endraw %}

i.e., use [Jekyll's tag link](https://jekyllrb.com/docs/templates/#links) and the name of the file.

## Episode Header

Each episode's [YAML][yaml] header must contain:

*   the episode's title
*   time estimates for teaching and exercises
*   motivating questions
*   lesson objectives
*   a summary of key points

These values are stored in the header so that [Jekyll][jekyll] will read them
and make them accessible in other pages as `site.episodes.the_episode.key`,
where `the_episode` is the particular episode
and `key` is the key in the [YAML][yaml] header.
This lets us do things like
list each episode's key questions in the syllabus on the lesson home page.

## Episode Structure

The episode layout template in `_layouts/episode.html` automatically creates
an introductory block that summarizes the lesson's teaching time,
exercise time,
key questions,
and objectives.
It also automatically creates a closing block that lists its key points.
In between,
authors should use only:

*   paragraphs
*   images
*   tables
*   ordered and unordered lists
*   code samples (described below).
*   special blockquotes (described below)

Authors should *not* use:

*   sub-headings
*   HTML layout (e.g., `div` elements).


> ## Linking section IDs
>
> In the HTML output each header of a section, code sample, exercise will be associated with an unique ID (the rules of
> the ID generation are given in kramdown [documentation](https://kramdown.gettalong.org/converter/html.html#auto-ids),
> but it is easier to look for them directly in the page sources).
> These IDs can be used to easily link to the section by attaching the hash (`#`) followed by the ID to the page's URL
> (like [this](#linking-section-ids)). For example, the instructor might copy the link to
> the etherpad, so that the lesson opens in learners' web browser directly at the right spot.
{: .callout}

## Formatting Code

Inline code fragments are formatted using back-quotes.
Longer code blocks are formatted by opening and closing the block with `~~~` (three tildes),
with a class specifier after the block:

{% raw %}
    ~~~
    for thing in collection:
        do_something
    ~~~
    {: .source}
{% endraw %}

which is rendered as:

~~~
for thing in collection:
    do_something
~~~
{: .source}

The class specified at the bottom using an opening curly brace and colon,
the class identifier with a leading dot,
and a closing curly brace.
The [template]({{ site.template_repo }}) provides three styles for code blocks:

~~~
.source: program source.
~~~
{: .source}

~~~
.output: program output.
~~~
{: .output}

~~~
.error: error messages.
~~~
{: .error}

### Syntax Highlighting

The following styles like `.source`, but include syntax highlighting for the
specified language.
Please use them where possible to indicate the type of source being displayed,
and to make code easier to read.

`.language-bash`: Bash shell commands:

~~~
echo "Hello World"
~~~
{: .language-bash}

`.html`: HTML source:

~~~
<html>
<body>
<em>Hello World</em>
</body>
</html>
~~~
{: .html}

`.language-make`: Makefiles:

~~~
all:
    g++ main.cpp hello.cpp -o hello
~~~
{: .language-make}

`.language-matlab`: MATLAB source:

~~~
disp('Hello, world!')
~~~
{: .language-matlab}

`.language-python`: Python source:

~~~
print("Hello World")
~~~
{: .language-python}

`.language-r`: R source:

~~~
cat("Hello World")
~~~
{: .language-r}

`.language-sql`: SQL source:

~~~
CREATE PROCEDURE HelloWorld AS
PRINT 'Hello, world!'
RETURN (0)
~~~
{: .language-sql}



## Special Blockquotes

We use blockquotes to group headings and text
rather than wrapping them in `div` elements.
in order to avoid confusing [Jekyll][jekyll]'s parser
(which sometimes has trouble with Markdown inside HTML).
Each special blockquote must started with a level-2 header,
but may contain anything after that.
For example,
a callout is formatted like this:

~~~
> ## Callout Title
>
> text
> text
> text
>
> ~~~
> code
> ~~~
> {: .source}
{: .callout}
~~~
{: .source}

(Note the empty lines within the blockquote after the title and before the code block.)
This is rendered as:

> ## Callout Title
>
> text
> text
> text
>
> ~~~
> code
> ~~~
> {: .source}
{: .callout}

The [lesson template]({{ site.template_repo }}) defines styles
for the following special blockquotes:

<div class="row">
  <div class="col-md-6" markdown="1">

> ## `.callout`
>
> An aside or other comment.
{: .callout}

> ## `.challenge`
>
> An exercise.
{: .challenge}

> ## `.checklist`
>
> Checklists.
{: .checklist}

> ## `.discussion`
>
> Discussion questions.
{: .discussion}

> ## `.keypoints`
>
> Key points of an episode.
{: .keypoints}

  </div>
  <div class="col-md-6" markdown="1">

> ## `.objectives`
>
> Episode objectives.
{: .objectives}

> ## `.prereq`
>
> Prerequisites.
{: .prereq}

> ## `.solution`
>
> Exercise solution.
{: .solution}

> ## `.testimonial`
>
> A laudatory quote from a user.
{: .testimonial}

  </div>
</div>

Note that `.challenge` and `.discussion` have the same color but different icons.
Note also that one other class, `.quotation`,
is used to mark actual quotations
(the original purpose of the blockquote element).
This does not add any styling,
but is used to prevent the checking tools from complaining about a missing class.

Most authors will only use `.callout`, `.challenge`, and `.prereq`,
as the others are automatically generated by the template.
Note that `.prereq` is meant for describing things
that learners should know before starting this lesson;
setup instructions do not have a particular style,
but are instead put on the `setup.md` page.

Note also that solutions are nested inside exercises as shown below:

~~~
> ## Challenge Title
>
> This is the body of the challenge.
>
> ~~~
> it may include some code
> ~~~
> {: .source}
>
> > ## Solution
> >
> > This is the body of the solution.
> >
> > ~~~
> > it may also include some code
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}
~~~
{: .source}

The double indentation is annoying to edit,
but the alternatives we considered and discarded are worse:

1.  Use HTML `<div>` elements for the challenges.
    Most people dislike mixing HTML and Markdown,
    and experience shows that it's all too easy to confuse Jekyll's Markdown parser.

2.  Put solutions immediately after challenges rather than inside them.
    This is simpler to edit,
    but clutters up the page
    and makes it harder for tools to tell which solutions belong to which exercises.

{% include links.md %}
