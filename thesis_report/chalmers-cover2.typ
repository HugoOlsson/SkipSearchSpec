// chalmers-cover.typ
//
// Typst port of the Chalmers / University of Gothenburg master's thesis
// cover template.
// Original LaTeX by David Frisk (2016), Jakob Jarmar (2016),
// Birgit Grohe (2017+), with adjustments by Gustav Örtenberg (2019).
//
// Usage:
//   #import "chalmers-cover.typ": cover-pages
//   #cover-pages(
//     title: "My Thesis Title",
//     subtitle: "An optional subtitle",
//     author: "Firstname Lastname",
//     supervisor: "Supervisor Name, Department",
//     examiner: "Examiner Name, Department",
//   )
//   // ... rest of your thesis ...

#let cover-pages(
  title: "Thesis Title",
  subtitle: none,
  author: "Name Familyname",
  program: "Computer science and engineering",
  supervisor: "Name, Department",
  advisor: none,                                   // optional
  examiner: "Name, Department",
  year: datetime.today().year(),
  background-image: "figure/auxiliary/old frontpages/frontpage_eng.pdf",
  logo: "figure/auxiliary/old frontpages/logo_eng.pdf",
  cover-figure: none,                              // optional image on cover
  cover-caption: none,                             // optional caption on imprint page
) = {
  let sans  = ("Arial", "Liberation Sans", "Helvetica")
  let serif = ("New Computer Modern", "Latin Modern Roman", "TeX Gyre Termes")

  // ------------------------------------------------------------------
  // 1. COVER PAGE
  // ------------------------------------------------------------------
  page(
    paper: "a4",
    margin: (top: 3cm, bottom: 1cm, left: 2.25cm, right: 2.25cm),
    background: place(
      top + left,
      dx: -4mm,
      dy: 0mm,
      image(background-image),
    ),
  )[
    #set text(font: sans, size: 12pt, lang: "en")

    #v(1fr)

    #if cover-figure != none {
      align(center, cover-figure)
      v(1cm)
    }

    #text(size: 24.88pt, weight: "bold", title) \
    #v(0.1cm)
    #if subtitle != none [
      #text(size: 17.28pt, subtitle) \
      #v(0.5cm)
    ]
    Master's thesis in #program
    #v(1cm)
    #text(size: 17.28pt, author)
    #v(2.9cm)

    Department of Computer Science and Engineering \
    #smallcaps[Chalmers University of Technology] \
    Gothenburg, Sweden #year
  ]

  // ------------------------------------------------------------------
  // 2. BACK OF COVER (blank)
  // ------------------------------------------------------------------
  page(margin: 2.5cm)[]

  // ------------------------------------------------------------------
  // 3. TITLE PAGE
  // ------------------------------------------------------------------
  page(margin: 2.5cm)[
    #set text(font: serif, size: 12pt)
    #align(center)[
      #text(size: 14.4pt, smallcaps[Master's thesis #year])

      #v(4cm)

      #text(size: 17.28pt, weight: "bold", title)

      #v(1cm)

      #if subtitle != none [
        #text(size: 14.4pt, subtitle)
        #v(1cm)
      ]

      #text(size: 14.4pt, author)

      #v(1fr)

      #image(logo, width: 25%)

      #v(5mm)

      Department of Computer Science and Engineering \
      #smallcaps[Chalmers University of Technology] \
      Gothenburg, Sweden #year
    ]
  ]

  // ------------------------------------------------------------------
  // 4. IMPRINT PAGE
  // ------------------------------------------------------------------
  page(margin: 2.5cm)[
    #set text(size: 12pt)
    #v(4.5cm)

    #title \
    #if subtitle != none [#subtitle \ ]
    #author

    #v(1cm)

    © #author, #year.

    #v(1cm)

    Supervisor: #supervisor \
    #if advisor != none [Advisor: #advisor \ ]
    Examiner: #examiner

    #v(1cm)

    Master's Thesis #year \
    Department of Computer Science and Engineering \
    Chalmers University of Technology \
    SE-412 96 Gothenburg \
    Telephone +46 31 772 1000

    #v(1fr)

    #if cover-caption != none [
      Cover: #cover-caption

    ]
    //Typeset in Typst \
    Gothenburg, Sweden #year
  ]
}
