"""
Database Explorer Router

Provides endpoints to inspect the database contents — useful for
debugging and understanding what data is stored after analysis sessions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/db/tables",
    summary="List all database tables and their columns",
)
async def list_tables():
    """Return every table name along with its column definitions."""
    async with engine.connect() as conn:
        # Use SQLAlchemy inspector via run_sync
        def _inspect(sync_conn):
            insp = inspect(sync_conn)
            tables = {}
            for table_name in insp.get_table_names():
                columns = []
                for col in insp.get_columns(table_name):
                    columns.append(
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col.get("nullable", True),
                            "default": str(col.get("default")) if col.get("default") else None,
                            "primary_key": col.get("autoincrement", False) == True
                            or col["name"] == "id",
                        }
                    )

                # Foreign keys
                fks = []
                for fk in insp.get_foreign_keys(table_name):
                    fks.append(
                        {
                            "columns": fk["constrained_columns"],
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"],
                        }
                    )

                # Indexes
                indexes = []
                for idx in insp.get_indexes(table_name):
                    indexes.append(
                        {
                            "name": idx["name"],
                            "columns": idx["column_names"],
                            "unique": idx.get("unique", False),
                        }
                    )

                tables[table_name] = {
                    "columns": columns,
                    "foreign_keys": fks,
                    "indexes": indexes,
                }
            return tables

        tables = await conn.run_sync(_inspect)
    return {"tables": tables}


@router.get(
    "/db/stats",
    summary="Get row counts and basic statistics for all tables",
)
async def db_stats(db: AsyncSession = Depends(get_db)):
    """Return row counts and useful aggregates."""
    stats = {}

    # Sessions
    result = await db.execute(text("SELECT COUNT(*) FROM sessions"))
    total_sessions = result.scalar_one()

    status_result = await db.execute(
        text("SELECT status, COUNT(*) FROM sessions GROUP BY status")
    )
    status_counts = {row[0]: row[1] for row in status_result.all()}

    stats["sessions"] = {
        "total": total_sessions,
        "by_status": status_counts,
    }

    # Students
    result = await db.execute(text("SELECT COUNT(*) FROM students"))
    total_students = result.scalar_one()
    stats["students"] = {"total": total_students}

    # Posture records
    result = await db.execute(text("SELECT COUNT(*) FROM posture_records"))
    total_records = result.scalar_one()

    posture_result = await db.execute(
        text("SELECT posture, COUNT(*) FROM posture_records GROUP BY posture ORDER BY COUNT(*) DESC")
    )
    posture_counts = {row[0]: row[1] for row in posture_result.all()}

    avg_conf = await db.execute(
        text("SELECT AVG(confidence) FROM posture_records")
    )
    avg_confidence = avg_conf.scalar_one()

    stats["posture_records"] = {
        "total": total_records,
        "by_posture": posture_counts,
        "average_confidence": round(float(avg_confidence), 4) if avg_confidence else 0.0,
    }

    # Database file size (SQLite only)
    try:
        result = await db.execute(text("PRAGMA page_count"))
        page_count = result.scalar_one()
        result = await db.execute(text("PRAGMA page_size"))
        page_size = result.scalar_one()
        stats["database"] = {
            "size_mb": round((page_count * page_size) / (1024 * 1024), 2),
            "type": "SQLite",
        }
    except Exception:
        stats["database"] = {"type": "PostgreSQL or other"}

    return stats


@router.get(
    "/db/browse/{table_name}",
    summary="Browse rows in a table with pagination",
)
async def browse_table(
    table_name: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Rows per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated view of any table's contents.

    Allowed tables: sessions, students, posture_records
    """
    allowed_tables = {"sessions", "students", "posture_records"}
    if table_name not in allowed_tables:
        return {"error": f"Table '{table_name}' not allowed. Choose from: {sorted(allowed_tables)}"}

    offset = (page - 1) * page_size

    # Count total rows
    count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
    total = count_result.scalar_one()

    # Fetch rows
    result = await db.execute(
        text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT :limit OFFSET :offset"),
        {"limit": page_size, "offset": offset},
    )

    columns = list(result.keys())
    rows = [dict(zip(columns, row)) for row in result.all()]

    # Convert non-serializable types to strings
    for row in rows:
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                row[key] = value.isoformat()
            elif isinstance(value, bytes):
                row[key] = value.hex()

    return {
        "table": table_name,
        "columns": columns,
        "rows": rows,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }
