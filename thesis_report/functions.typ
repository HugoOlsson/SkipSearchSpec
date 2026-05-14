#let probing-table(
  title: none,
  model-name: none,
  commit-id: none,
  windows: none,
  window-length: none,
  total-clusters: none,
  rows: none,

  top-k-label: [*top k clusters*],
  pct-label: [*% of all clusters probed*],
  top1-label: [*top1 match rate*],
  top3-label: [*top3 match rate*],
  caption: [Match-rate sweep over `top_k_clusters`.],

  table-width: 100%,
  outer-inset: 10pt,
  radius: 6pt,

  base-size: 9pt,
  title-size: 13pt,
  model-size: 8.5pt,
  meta-size: 8pt,
  header-size: 8pt,
  body-size: 8pt,
  caption-size: 8pt,

  cell-inset: (x: 6pt, y: 4pt),

  table-border: rgb("#d0d7de"),
  header-bg: rgb("#f1f5f9"),
  title-color: rgb("#111827"),
  muted: rgb("#64748b"),
  fill-color: white,

  columns: (0.95fr, 1.2fr, 1fr, 1fr),
  align-cols: (left, right, right, right),
) = {
  let body-cell(x) = [#text(size: body-size)[#x]]
  let header-cell(x) = table.cell(fill: header-bg)[
    #text(size: header-size)[#x]
  ]

  figure(
    block(
      width: table-width,
      inset: outer-inset,
      radius: radius,
      stroke: 0.8pt + table-border,
      fill: fill-color,
    )[
      #set text(size: base-size)

      #align(center)[
        #text(size: title-size, weight: "bold", fill: title-color)[#title] \
        #text(size: model-size, weight: "medium", fill: title-color)[#model-name]

        #text(size: meta-size, fill: muted)[
          commit=#commit-id · windows=#windows · window_length=#window-length · total_clusters=#total-clusters
        ]
      ]

      #table(
        columns: columns,
        align: align-cols,
        inset: cell-inset,
        stroke: table-border,

        table.header(
          header-cell(top-k-label),
          header-cell(pct-label),
          header-cell(top1-label),
          header-cell(top3-label),
        ),

        ..rows.map(row => (
          body-cell(row.at(0)),
          body-cell(row.at(1)),
          body-cell(row.at(2)),
          body-cell(row.at(3)),
        )).flatten()
      )
    ],
    caption: caption,
  )
}