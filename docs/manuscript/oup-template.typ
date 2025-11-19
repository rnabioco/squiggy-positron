// Oxford University Press Journal Template (Typst Version)
// Converted from OUP LaTeX authoring template

#let oup-article(
  title: "",
  short-title: none,
  authors: (),
  affiliations: (),
  corresponding: none,
  abstract: [],
  keywords: (),
  journal-title: "Journal Title Here",
  doi: "DOI HERE",
  copyright-year: 2022,
  pub-year: 2019,
  access-date: "Day Month Year",
  paper-type: "Paper",
  first-page: 1,
  received: none,
  revised: none,
  accepted: none,
  body
) = {
  // Page setup
  set page(
    paper: "us-letter",
    margin: (x: 1.5in, y: 1in),
    header: locate(loc => {
      if counter(page).at(loc).first() > 1 [
        #set text(9pt)
        #grid(
          columns: (1fr, 1fr),
          align: (left, right),
          [#emph[#authors.map(a => a.name).join(", et al." if authors.len() > 3 else " and ")]],
          [Page #counter(page).display()]
        )
        #line(length: 100%, stroke: 0.5pt)
      ]
    }),
    footer: locate(loc => {
      if counter(page).at(loc).first() == 1 [
        #set text(8pt)
        #align(center)[
          #journal-title | #doi
        ]
      ]
    })
  )

  // Text settings
  set text(
    font: "New Computer Modern",
    size: 10pt,
    lang: "en"
  )

  set par(
    justify: true,
    leading: 0.65em,
    first-line-indent: 1.5em
  )

  // Heading settings
  set heading(numbering: "1.1")

  show heading.where(level: 1): it => {
    set text(12pt, weight: "bold")
    block(above: 14pt, below: 10pt, it)
  }

  show heading.where(level: 2): it => {
    set text(11pt, weight: "bold")
    block(above: 12pt, below: 8pt, it)
  }

  show heading.where(level: 3): it => {
    set text(10pt, weight: "bold")
    block(above: 10pt, below: 6pt, it)
  }

  show heading.where(level: 4): it => {
    set text(10pt, weight: "bold", style: "italic")
    block(above: 8pt, below: 4pt)[#it.body. ]
  }

  // Title
  align(center)[
    #block(above: 12pt, below: 12pt)[
      #text(18pt, weight: "bold")[#title]
    ]
  ]

  // Authors and affiliations
  align(center)[
    #block(above: 8pt, below: 8pt)[
      #for (i, author) in authors.enumerate() [
        #author.name#super[#author.affiliations.join(",")]#if i < authors.len() - 1 [, ] else if author.corresponding [#super[*]]
      ]
    ]
  ]

  // Affiliations
  align(left)[
    #block(above: 8pt, below: 8pt)[
      #set text(9pt)
      #for (i, affil) in affiliations.enumerate() [
        #super[#str(i + 1)]#affil \
      ]
      #if corresponding != none [
        #super[*]Corresponding author. #link("mailto:" + corresponding)[#corresponding] \
      ]
    ]
  ]

  // Dates
  if received != none or revised != none or accepted != none [
    #block(above: 8pt, below: 8pt)[
      #set text(9pt, style: "italic")
      #if received != none [Received: #received]
      #if revised != none [ | Revised: #revised]
      #if accepted != none [ | Accepted: #accepted]
    ]
  ]

  // Abstract
  block(above: 12pt, below: 12pt)[
    #set par(first-line-indent: 0pt)
    #text(weight: "bold")[Abstract] \
    #abstract
  ]

  // Keywords
  if keywords.len() > 0 [
    #block(above: 8pt, below: 12pt)[
      #set par(first-line-indent: 0pt)
      #text(weight: "bold")[Keywords: ]#keywords.join(", ")
    ]
  ]

  // Horizontal rule before main content
  line(length: 100%, stroke: 1pt)
  v(12pt)

  // Main body
  body
}

// Theorem environments
#let theorem(body, name: none) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #set text(style: "italic")
    #text(weight: "bold", style: "normal")[
      Theorem#if name != none [ (#name)].]
    #h(0.5em)#body
  ]
}

#let proposition(body, name: none) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #set text(style: "italic")
    #text(weight: "bold", style: "normal")[
      Proposition#if name != none [ (#name)].]
    #h(0.5em)#body
  ]
}

#let definition(body, name: none) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #text(weight: "bold")[
      Definition#if name != none [ (#name)].]
    #h(0.5em)#body
  ]
}

#let example(body) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #set text(style: "italic")
    #text(weight: "bold", style: "normal")[Example.]
    #h(0.5em)#body
  ]
}

#let remark(body) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #set text(style: "italic")
    #text(weight: "bold", style: "normal")[Remark.]
    #h(0.5em)#body
  ]
}

#let proof(body, name: none) = {
  block(above: 10pt, below: 10pt, inset: 0pt)[
    #text(style: "italic")[
      Proof#if name != none [ of #name].]
    #h(0.5em)#body
    #h(1fr)
    $square$
  ]
}

// Biography section
#let biography(body, image: none, name: "") = {
  block(above: 12pt, below: 12pt)[
    #if image != none [
      #grid(
        columns: (77pt, 1fr),
        column-gutter: 10pt,
        image,
        [#text(weight: "bold")[#name] #body]
      )
    ] else [
      #text(weight: "bold")[#name] #body
    ]
  ]
}

// Main document template
#show: oup-article.with(
  title: "Article Title",
  short-title: "Short Article Title",
  authors: (
    (name: "First Author", affiliations: (1,), corresponding: true),
    (name: "Second Author", affiliations: (2,)),
    (name: "Third Author", affiliations: (3,)),
    (name: "Fourth Author", affiliations: (3,)),
    (name: "Fifth Author", affiliations: (4,)),
  ),
  affiliations: (
    "Department, Organization, Street, Postcode, State, Country",
    "Department, Organization, Street, Postcode, State, Country",
    "Department, Organization, Street, Postcode, State, Country",
    "Department, Organization, Street, Postcode, State, Country",
  ),
  corresponding: "email-id.com",
  abstract: [
    Abstracts must be able to stand alone and so cannot contain citations to the paper's references, equations, etc. An abstract must consist of a single paragraph and be concise. Because of online formatting, abstracts must appear as plain as possible.
  ],
  keywords: ("keyword1", "Keyword2", "Keyword3", "Keyword4"),
  journal-title: "Journal Title Here",
  doi: "DOI HERE",
  copyright-year: 2022,
  pub-year: 2019,
  received: "Date 0 Year",
  revised: "Date 0 Year",
  accepted: "Date 0 Year",
)

= Introduction <intro>

The introduction introduces the context and summarizes the manuscript. It is importantly to clearly state the contributions of this piece of work. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

This is an example of a new paragraph with a numbered footnote#footnote[#link("https://data.gov.uk/")] and a second footnote marker.#footnote[Example of footnote text.]

= This is an example for first level head - section head <sec2>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum (refer @sec5).

== This is an example for second level head - subsection head <subsec1>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

=== This is an example for third level head - subsubsection head <subsubsec1>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

==== This is an example for fourth level head - paragraph head

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

= This is an example for first level head <sec3>

== This is an example for second level head - subsection head <subsec2>

=== This is an example for third level head - subsubsection head <subsubsec2>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

= Equations <sec4>

Equations in Typst can either be inline or set as display equations. For inline equations use the `$...$` syntax. Eg: the equation $H psi = E psi$ is written via `$H psi = E psi$`.

For display equations one can use the equation environment:

$ norm(tilde(X)(k))^2 <= (sum_(i=1)^p norm(tilde(Y)_i (k))^2 + sum_(j=1)^q norm(tilde(Z)_j (k))^2)/(p+q) $ <eq1>

where,

$ D_mu &= partial_mu - i g (lambda^a)/2 A^a_mu \
F^a_(mu nu) &= partial_mu A^a_nu - partial_nu A^a_mu + g f^(a b c) A^b_mu A^a_nu $ <eq2>

$ Y_infinity = ((m)/"GeV")^(-3) [1 + (3 ln(m"/"GeV))/15 + (ln(c_2"/"5))/15] $

The Typst equivalent supports mathematical symbols like $bb(R)$, $cal(R)$ for blackboard bold and calligraphic letters (refer @subsubsec3).

= Tables <sec5>

Tables can be inserted via the table environment. To put footnotes inside tables one can add them below the table itself (refer @tab1 and @tab2).

#figure(
  table(
    columns: 4,
    stroke: (x, y) => (
      top: if y == 0 { 1pt } else if y == 1 { 0.5pt } else { 0pt },
      bottom: if y == 3 { 1pt } else { 0pt },
    ),
    align: left,
    [*column 1*], [*column 2*], [*column 3*], [*column 4*],
    [row 1], [data 1], [data 2], [data 3],
    [row 2], [data 4], [data 5#super[1]], [data 6],
    [row 3], [data 7], [data 8], [data 9#super[2]],
  ),
  caption: [Caption text],
  kind: table
) <tab1>

#block(above: 4pt)[
  #set text(8pt)
  Source: This is an example of table footnote. \
  #super[1] Example for a first table footnote. \
  #super[2] Example for a second table footnote.
]

#figure(
  table(
    columns: 7,
    stroke: (x, y) => (
      top: if y == 0 { 1pt } else if y == 2 { 0.5pt } else { 0pt },
      bottom: if y == 3 { 1pt } else { 0pt },
    ),
    align: (left, center, center, center, center, center, center),
    [], table.cell(colspan: 3)[*Element 1#super[1]*], table.cell(colspan: 3)[*Element 2#super[2]*],
    [*Project*], [*Energy*], [$sigma_"calc"$], [$sigma_"expt"$], [*Energy*], [$sigma_"calc"$], [$sigma_"expt"$],
    [Element 3], [990 A], [1168], [$1547 plus.minus 12$], [780 A], [1166], [$1239 plus.minus 100$],
    [Element 4], [500 A], [961], [$922 plus.minus 10$], [900 A], [1268], [$1092 plus.minus 40$],
  ),
  caption: [Example of a lengthy table which is set to full textwidth.],
  kind: table
) <tab2>

#block(above: 4pt)[
  #set text(8pt)
  Note: This is an example of table footnote. \
  #super[1] Example for a first table footnote. \
  #super[2] Example for a second table footnote.
]

= Figures <sec6>

Figures can be inserted via the normal figure environment as shown in the below example:

#figure(
  rect(width: 213pt, height: 37pt, fill: rgb(20%, 20%, 20%)),
  caption: [This is a widefig. This is an example of a long caption this is an example of a long caption this is an example of a long caption this is an example of a long caption],
) <fig1>

#figure(
  rect(width: 438pt, height: 74pt, fill: rgb(20%, 20%, 20%)),
  caption: [This is a widefig. This is an example of a long caption this is an example of a long caption this is an example of a long caption this is an example of a long caption],
) <fig2>

= Algorithms, Program codes and Listings <sec7>

For algorithms in Typst, you can use packages like `algo` or create custom formatting:

#figure(
  block(
    width: 100%,
    inset: 10pt,
    stroke: 1pt,
    [
      *Algorithm 1:* Calculate $y = x^n$ \
      *Require:* $n >= 0 or x != 0$ \
      *Ensure:* $y = x^n$ \
      1. $y <- 1$ \
      2. *If* $n < 0$ *then* \
      #h(1em) 3. $X <- 1"/"x$ \
      #h(1em) 4. $N <- -n$ \
      5. *Else* \
      #h(1em) 6. $X <- x$ \
      #h(1em) 7. $N <- n$ \
      8. *While* $N != 0$ *do* \
      #h(1em) 9. *If* $N$ is even *then* \
      #h(2em) 10. $X <- X times X$ \
      #h(2em) 11. $N <- N"/"2$ \
      #h(1em) 12. *Else* [$N$ is odd] \
      #h(2em) 13. $y <- y times X$ \
      #h(2em) 14. $N <- N - 1$
    ]
  ),
  caption: [Calculate $y = x^n$],
  kind: "algorithm"
) <algo1>

For code listings:

```pascal
for i:=maxint to 0 do
begin
  { do nothing }
end;
Write('Case insensitive ');
Write('Pascal keywords.');
```

= Cross referencing <sec8>

Environments such as figure, table, and equation can have a label declared via the `<#label>` syntax. One can then use the `@label` syntax to cross-reference them. As an example, consider the label declared for @fig1.

= Lists <sec9>

Lists in Typst can be numbered or bulleted.

+ This is the 1st item

+ Numbered lists and bulleted lists
  a. Second level numbered list

  b. Second level numbered list
    i. Third level numbered list

    ii. Third level numbered list

  c. Second level numbered list

+ Numbered lists continue

Bulleted lists:

- First level bulleted list. This is the 1st item

- First level bulleted list
  - Second level dashed list

  - Second level dashed list

- First level bulleted list

= Examples for theorem-like environments <sec10>

#theorem(name: "Theorem subhead")[
  Example theorem text. Example theorem text. Example theorem text. Example theorem text. Example theorem text. Example theorem text.
] <thm1>

#proposition[
  Example proposition text. Example proposition text. Example proposition text.
]

#example[
  Phasellus adipiscing semper elit. Proin fermentum massa ac quam. Sed diam turpis, molestie vitae, placerat a, molestie nec, leo.
]

#remark[
  Phasellus adipiscing semper elit. Proin fermentum massa ac quam.
]

#definition(name: "Definition sub head")[
  Example definition text. Example definition text. Example definition text.
]

#proof[
  Example for proof text. Example for proof text. Example for proof text.
]

#proof(name: "Theorem " + link(label("thm1"))[1])[
  Example for proof text. Example for proof text. Example for proof text.
]

For a quote environment:

#quote[
  Quoted text example. Aliquam porttitor quam a lacus. Praesent vel arcu ut tortor cursus volutpat.
]

= Conclusion

Some Conclusions here.

#heading(numbering: none)[Appendix]

= Section title of first appendix <sec11>

Nam dui ligula, fringilla a, euismod sodales, sollicitudin vel, wisi. Morbi auctor lorem non justo.

== Subsection title of first appendix <subsec4>

Nam dui ligula, fringilla a, euismod sodales, sollicitudin vel, wisi.

=== Subsubsection title of first appendix <subsubsec3>

Example for an unnumbered figure:

#figure(
  rect(width: 85pt, height: 92pt, fill: rgb(20%, 20%, 20%)),
  numbering: none
)

= Section title of second appendix <sec12>

Example for an equation inside the appendix:

$ p &= (gamma^2 - (n_C - 1)H)/((n_C - 1) + H - 2gamma) \
theta &= ((gamma - H)^2 (gamma - n_C - 1)^2)/((n_C - 1 + H - 2gamma)^2) $

#heading(numbering: none)[Competing interests]
No competing interest is declared.

#heading(numbering: none)[Author contributions statement]

Must include all authors, identified by initials, for example:
S.R. and D.A. conceived the experiment(s), S.R. conducted the experiment(s), S.R. and D.A. analysed the results. S.R. and D.A. wrote and reviewed the manuscript.

#heading(numbering: none)[Acknowledgments]
The authors thank the anonymous reviewers for their valuable suggestions. This work is supported in part by funds from the National Science Foundation (NSF: \# 1636933 and \# 1920920).

#heading(numbering: none)[References]

#bibliography("references.bib", style: "ieee")

// Biography sections
#heading(numbering: none)[Author Biographies]

#biography(
  image: rect(width: 77pt, height: 77pt, fill: rgb(20%, 20%, 20%)),
  name: "Author Name."
)[
  This is sample author biography text. This is sample author biography text this is sample author biography text this is sample author biography text.
]

#biography(name: "Author Name.")[
  This is sample author biography text this is sample author biography text this is sample author biography text.
]
