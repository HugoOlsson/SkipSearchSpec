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
  background-image: "figure/auxiliary/frontpage_gu_eng_vec_m2.pdf",
  logo: "figure/auxiliary/ChGULogoHog.pdf",
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
    #set text(font: sans, lang: "en")

    #v(1fr)

    #if cover-figure != none {
      align(center, cover-figure)
      v(1cm)
    }

    #text(size: 24pt, weight: "bold", title) \
    #v(0.5cm)
    #if subtitle != none [
      #text(size: 14pt, subtitle) \
      #v(0.5cm)
    ]
    Master's thesis in #program
    #v(1cm)
    #text(size: 14pt, author)
    #v(2.9cm)

    Department of Computer Science and Engineering \
    #smallcaps[Chalmers University of Technology] \
    #smallcaps[University of Gothenburg] \
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
    #set text(font: serif)
    #align(center)[
      #text(size: 14pt, smallcaps[Master's thesis #year])

      #v(4cm)

      #text(size: 17pt, weight: "bold", title)

      #v(1cm)

      #if subtitle != none [
        #text(size: 14pt, subtitle)
        #v(1cm)
      ]

      #text(size: 14pt, author)

      #v(1fr)

      #image(logo, width: 25%)

      #v(5mm)

      Department of Computer Science and Engineering \
      #smallcaps[Chalmers University of Technology] \
      #smallcaps[University of Gothenburg] \
      Gothenburg, Sweden #year
    ]
  ]

  // ------------------------------------------------------------------
  // 4. IMPRINT PAGE
  // ------------------------------------------------------------------
  page(margin: 2.5cm)[
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
    Chalmers University of Technology and University of Gothenburg \
    SE-412 96 Gothenburg \
    Telephone +46 31 772 1000

    #v(1fr)

    #if cover-caption != none [
      Cover: #cover-caption

    ]
    Typeset in Typst \
    Gothenburg, Sweden #year
  ]
}
