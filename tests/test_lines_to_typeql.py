from typedb_kgqa.graphrag.construction import lines_to_typeql


def test_entity():
    lines = "entity person:xawery-zulawski"
    result = lines_to_typeql(lines)
    assert 'put $person-xawery-zulawski isa entity-node, has node-label "person:xawery-zulawski";' in result


def test_property_string():
    lines = 'entity person:xawery-zulawski\nproperty person:xawery-zulawski name:xawery-zulawski "xawery zulawski"'
    result = lines_to_typeql(lines)
    assert 'has string-property "xawery zulawski"' in result
    assert 'has node-label "person:xawery-zulawski::name:xawery-zulawski"' in result
    assert 'links (owner: $person-xawery-zulawski)' in result


def test_property_date():
    lines = "entity person:x\nproperty person:x date-of-birth:1971-12-22 1971-12-22"
    result = lines_to_typeql(lines)
    assert "has date-property 1971-12-22" in result


def test_property_numeric():
    lines = "entity person:x\nproperty person:x height:180 180"
    result = lines_to_typeql(lines)
    assert "has numeric-property 180" in result


def test_property_boolean():
    lines = "entity person:x\nproperty person:x alive:true true"
    result = lines_to_typeql(lines)
    assert "has boolean-property true" in result


def test_relation():
    lines = "entity person:a\nentity person:b\nrelation person:a mother-of:a:b person:b"
    result = lines_to_typeql(lines)
    assert 'has node-label "mother-of:a:b"' in result
    assert "links (related: $person-a, related: $person-b)" in result


def test_source_generates_meta_document():
    lines = "source Xawery Żuławski\nentity person:xawery-zulawski"
    result = lines_to_typeql(lines)
    assert 'isa meta-document, has meta-page-title "Xawery Żuławski"' in result


def test_source_links_knowledge_source():
    lines = "source Some Title\nentity person:x"
    result = lines_to_typeql(lines)
    assert "isa meta-knowledge-source" in result
    assert "knowledge: $person-x" in result


def test_no_source_no_knowledge_links():
    lines = "entity person:x"
    result = lines_to_typeql(lines)
    assert "meta-knowledge-source" not in result


def test_comments_and_blanks_skipped():
    lines = "# comment\n\nentity person:x\n# another comment"
    result = lines_to_typeql(lines)
    stmts = [s for s in result.splitlines() if s.strip()]
    assert len(stmts) == 1
    assert "person:x" in stmts[0]


def test_embed_fn_called_for_entity():
    def mock_embed(label):
        return "EMBED_" + label

    lines = "entity person:x"
    result = lines_to_typeql(lines, embed_fn=mock_embed)
    assert 'has embedding "EMBED_person:x"' in result


def test_embed_fn_called_for_relation():
    def mock_embed(label):
        return "EMBED_" + label

    lines = "entity person:a\nentity person:b\nrelation person:a rel:a:b person:b"
    result = lines_to_typeql(lines, embed_fn=mock_embed)
    assert 'has embedding "EMBED_rel:a:b"' in result


def test_embed_fn_called_for_property():
    def mock_embed(label):
        return "EMBED_" + label

    lines = 'entity person:x\nproperty person:x name:x "x"'
    result = lines_to_typeql(lines, embed_fn=mock_embed)
    assert 'has embedding "EMBED_person:x::name:x"' in result


def test_embed_fn_none_no_embeddings():
    lines = "entity person:x"
    result = lines_to_typeql(lines, embed_fn=None)
    assert "embedding" not in result


def test_multiple_sources_switches_current():
    lines = "source Doc A\nentity person:a\nsource Doc B\nentity person:b"
    result = lines_to_typeql(lines)
    ks_lines = [l for l in result.splitlines() if "meta-knowledge-source" in l]
    # person:a linked to Doc A, person:b linked to Doc B
    assert len(ks_lines) == 2
    assert "person-a" in ks_lines[0]
    assert "person-b" in ks_lines[1]
