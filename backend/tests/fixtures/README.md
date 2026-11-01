# Test fixtures

`answer_stream_*.json` are **hand-constructed** Messages API streams shaped after the
documented event and citation formats (search results + citations, streaming
`citations_delta`, server-side fallback blocks, refusal stop reason). They are not
recordings of live API traffic: no API key was available when they were written. Once a
key is configured, `make record-fixtures` replaces them with real recordings, and the
same tests must keep passing.
