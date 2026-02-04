"""Fetch schema from a TypeDB instance and output as a TypeQL define query."""

from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType


def fetch_schema(
    address: str = "localhost:1729",
    database: str = "default",
    username: str = "admin",
    password: str = "password",
    compact: bool = False,
) -> str:
    """
    Fetch schema from TypeDB and return as a TypeQL define query.

    Args:
        address: TypeDB server address.
        database: Database name.
        username: Username for authentication.
        password: Password for authentication.
        compact: If True, return a compact representation for smaller LLMs.

    Returns:
        A syntactically valid TypeQL define query string (or compact format).
    """
    with TypeDB.driver(address, Credentials(username, password), DriverOptions(is_tls_enabled=False)) as driver:
        with driver.transaction(database, TransactionType.READ) as tx:
            entities = _fetch_entities(tx)
            attributes = _fetch_attributes(tx)
            relations = _fetch_relations(tx)
            owns = _fetch_owns(tx)
            relates = _fetch_relates(tx)
            plays = _fetch_plays(tx)

    if compact:
        return _build_compact_schema(entities, attributes, relations, owns, relates, plays)
    else:
        return _build_define_query(entities, attributes, relations, owns, relates, plays)

def _get_query_as_rows(tx, query):
    return list(tx.query(query).resolve().as_concept_rows())

def _fetch_entities(tx) -> list[str]:
    """Fetch all entity type labels."""
    results = _get_query_as_rows(tx, "match entity $x;")
    return [r.get("x").as_entity_type().get_label() for r in results]


def _fetch_attributes(tx) -> list[str]:
    """Fetch all attribute type labels."""
    results = _get_query_as_rows(tx, "match attribute $x;")
    attrs = []
    for r in results:
        attr_type = r.get("x").as_attribute_type()
        label = attr_type.get_label()
        attrs.append(label)
    return attrs


def _fetch_relations(tx) -> list[str]:
    """Fetch all relation type labels."""
    results = _get_query_as_rows(tx, "match relation $x;")
    return [r.get("x").as_relation_type().get_label() for r in results]


def _fetch_owns(tx) -> list[tuple[str, str]]:
    """Fetch all owns declarations (owner, attribute)."""
    results = _get_query_as_rows(tx, "match $x owns $y; not { $x sub! $z; $z owns $y; };")
    owns = []
    for r in results:
        owner = r.get("x").as_type().get_label()
        attr = r.get("y").as_attribute_type().get_label()
        owns.append((owner, attr))
    return owns


def _fetch_relates(tx) -> list[tuple[str, str]]:
    """Fetch all relates declarations (relation, role)."""
    results = _get_query_as_rows(tx, "match $x relates $y; not { $x sub! $z; $z relates $y; };")
    relates = []
    for r in results:
        relation = r.get("x").as_relation_type().get_label()
        role = r.get("y").as_role_type().get_label().split(":")[-1]
        relates.append((relation, role))
    return relates


def _fetch_plays(tx) -> list[tuple[str, str]]:
    """Fetch all plays declarations (player, role)."""
    results = _get_query_as_rows(tx, "match $x plays $y; not { $x sub! $z; $z plays $y; };")
    plays = []
    for r in results:
        player = r.get("x").as_type().get_label()
        role = r.get("y").as_role_type().get_label().split(":")[-1]
        plays.append((player, role))
    return plays


def _build_define_query(
    entities: list[str],
    attributes: list[str],
    relations: list[str],
    owns: list[tuple[str, str]],
    relates: list[tuple[str, str]],
    plays: list[tuple[str, str]],
) -> str:
    """Build a TypeQL define query from the fetched schema components."""
    lines = ["define"]

    # Entities
    for entity in sorted(entities):
        lines.append(f"entity {entity};")

    # Attributes
    for attr in sorted(attributes):
        lines.append(f"attribute {attr};")

    # Relations
    for relation in sorted(relations):
        lines.append(f"relation {relation};")

    # Owns
    for owner, attr in sorted(owns):
        lines.append(f"{owner} owns {attr};")

    # Relates
    for relation, role in sorted(relates):
        lines.append(f"{relation} relates {role};")

    # Plays
    for player, role in sorted(plays):
        lines.append(f"{player} plays {role};")

    return "\n".join(lines)


def _build_compact_schema(
    entities: list[str],
    attributes: list[str],
    relations: list[str],
    owns: list[tuple[str, str]],
    relates: list[tuple[str, str]],
    plays: list[tuple[str, str]],
) -> str:
    """
    Build a compact functional schema representation.

    Format:
        entity person has name|age|gender
        relation parentage links (parent: person|..., child: person|...)
    """
    from collections import defaultdict

    lines = []
    # Output entities and relations
    lines.append(f"$var isa { ' | '.join(sorted(entities + relations))};")

    # Group owns by owner: {owner: [attr1, attr2, ...]}
    owns_by_owner = defaultdict(list)
    for owner, attr in owns:
        owns_by_owner[owner].append(attr)

    # Group relates by relation: {relation: [role1, role2, ...]}
    relates_by_rel = defaultdict(list)
    for relation, role in relates:
        relates_by_rel[relation].append(role)

    # Group plays by role: {role: [player1, player2, ...]}
    plays_by_role = defaultdict(list)
    for player, role in plays:
        plays_by_role[role].append(player)

    # Output ownerships
    lines.append("# Has")
    for owner in sorted(entities + relations):
        attrs = owns_by_owner.get(owner, [])
        if attrs:
            lines.append(f"{owner} has {' | '.join(sorted(attrs))};")

    lines.append("")
    lines.append("# Links")

    # Output links
    for relation in sorted(relations):
        roles = relates_by_rel.get(relation, [])
        if roles:
            role_parts = []
            for role in sorted(roles):
                players = plays_by_role.get(role, [])
                if players:
                    role_parts.append(f"{role}: {' | '.join(sorted(players))}")
                else:
                    role_parts.append(role)
            lines.append(f"{relation} links ({', '.join(role_parts)});")

    return "\n".join(lines)
