#!/usr/bin/env python3
import base64
import csv
import os
import sys
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import click
from openai import OpenAI
from pydantic import BaseModel, Field

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Gmail API
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Broad flight-related search, excluding common noise
DEFAULT_QUERY = (
    "( subject:(flight OR itinerary OR \"booking confirmation\" OR \"flight confirmation\" OR \"e-ticket\" OR eticket OR \"e ticket\") ) "
    "-subject:(hotel OR car OR baggage OR \"seat sale\")"
)

MIN_DATE = date(2014, 1, 1)
DEFAULT_OUTPUT = os.path.abspath("flights.csv")


@dataclass
class FlightRow:
    date_iso: str
    origin_iata: str
    destination_iata: str
    flight_number: str = ""
    airline: str = ""
    seat: str = ""
    cabin: str = ""
    confirmation_code: str = ""
    notes: str = ""


class AiPassenger(BaseModel):
    name: Optional[str] = None


class AiFlight(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD for the flight date")
    origin_iata: Optional[str] = Field(None, description="IATA/ICAO code or airport name")
    destination_iata: Optional[str] = Field(None, description="IATA/ICAO code or airport name")
    flight_number: Optional[str] = None
    airline: Optional[str] = None
    seat: Optional[str] = None
    cabin: Optional[str] = None
    confirmation_code: Optional[str] = None


class AiExtraction(BaseModel):
    is_flight_booked_by_user: bool
    rationale: Optional[str] = None
    passenger_names: List[AiPassenger] = Field(default_factory=list)
    flights: List[AiFlight] = Field(default_factory=list)


def build_gmail_service() -> Any:
    token_path = os.path.abspath("token.json")
    creds_path = os.path.abspath("credentials.json")
    creds: Optional[Credentials] = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def search_message_ids(service: Any, query: str) -> List[str]:
    message_ids: List[str] = []
    page_token: Optional[str] = None
    while True:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=query, pageToken=page_token, maxResults=500)
            .execute()
        )
        message_ids.extend([m["id"] for m in resp.get("messages", [])])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return message_ids


def get_message(service: Any, msg_id: str) -> Dict[str, Any]:
    return service.users().messages().get(userId="me", id=msg_id, format="full").execute()


def _iter_parts(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not payload:
        return
    parts = payload.get("parts")
    if parts:
        for part in parts:
            mime = part.get("mimeType", "")
            if mime.startswith("multipart/"):
                yield from _iter_parts(part)
            else:
                yield part
    else:
        yield payload


def _decode_body(body: Dict[str, Any]) -> bytes:
    data = body.get("data")
    if not data:
        return b""
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def extract_bodies(message: Dict[str, Any]) -> Tuple[str, str]:
    text_chunks: List[str] = []
    html_chunks: List[str] = []
    payload = message.get("payload", {})
    for part in _iter_parts(payload):
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        if not body:
            continue
        content = _decode_body(body).decode("utf-8", errors="ignore")
        if mime == "text/plain":
            text_chunks.append(content)
        elif mime == "text/html":
            html_chunks.append(content)
    return ("\n".join(text_chunks), "\n".join(html_chunks))


def ai_extract_rows(text: str, html: str, user_name: str) -> List[FlightRow]:
    client = OpenAI()
    system_prompt = (
        "You are extracting flight itineraries from an email text/html. Identify whether the email is a flight booked by the provided user and extract normalized flight records if so.\n"
        "- is_flight_booked_by_user MUST be true only if this email is a clear flight booking/eticket/itinerary confirmation for the user (not marketing, promos, fare sales, schedules, trains, or generic alerts).\n"
        "- Strong evidence required: passenger section naming the user, and/or booking reference (PNR) and confirmation/itinerary phrasing. If such evidence is missing, set is_flight_booked_by_user=false and return flights=[].\n"
        "- Normalize date to YYYY-MM-DD.\n"
        "- Airports MUST be standard IATA 3-letter codes (e.g., SFO, LHR). If only a full airport name is present, infer the IATA code from context if unambiguous; otherwise leave the field empty rather than a full name.\n"
        "- flight_number like AA123.\n"
        "- confirmation_code MUST be the confirmation code/PNR only (typically 5–8 alphanumeric, not purely numeric). NEVER return long numeric ticket numbers as confirmation_code. If only a ticket number is present, leave confirmation_code empty.\n"
        "- If the content appears promotional or informational without a confirmed booking, set is_flight_booked_by_user=false and flights=[]."
    )
    user_content = f"USER: {user_name}\n\nEMAIL_TEXT:\n{text[:100000]}\n\nEMAIL_HTML:\n{html[:100000]}"
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=AiExtraction,
        temperature=0,
    )
    data: AiExtraction = resp.choices[0].message.parsed  # type: ignore[assignment]
    if not data or not data.is_flight_booked_by_user:
        return []
    rows: List[FlightRow] = []
    for f in data.flights:
        if not (f.date and f.origin_iata and f.destination_iata):
            continue
        rows.append(
            FlightRow(
                date_iso=f.date,
                origin_iata=(f.origin_iata or "").upper(),
                destination_iata=(f.destination_iata or "").upper(),
                flight_number=(f.flight_number or "").upper(),
                airline=f.airline or "",
                seat=f.seat or "",
                cabin=f.cabin or "",
                confirmation_code=f.confirmation_code or "",
                notes="",
            )
        )
    return rows


def filter_by_date(rows: List[FlightRow], start_date: Optional[date], end_date: Optional[date]) -> List[FlightRow]:
    kept: List[FlightRow] = []
    for r in rows:
        try:
            d = datetime.strptime(r.date_iso, "%Y-%m-%d").date()
        except Exception:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        kept.append(r)
    return kept


def _merge_rows(existing: FlightRow, new: FlightRow) -> None:
    if not existing.origin_iata and new.origin_iata:
        existing.origin_iata = new.origin_iata
    if not existing.destination_iata and new.destination_iata:
        existing.destination_iata = new.destination_iata
    if not existing.flight_number and new.flight_number:
        existing.flight_number = new.flight_number
    if not existing.airline and new.airline:
        existing.airline = new.airline
    if not existing.seat and new.seat:
        existing.seat = new.seat
    if not existing.cabin and new.cabin:
        existing.cabin = new.cabin
    if not existing.confirmation_code and new.confirmation_code:
        existing.confirmation_code = new.confirmation_code
    if not existing.notes and new.notes:
        existing.notes = new.notes


def write_csv(rows: List[FlightRow], path: str) -> None:
    headers = [
        "Date",
        "From",
        "To",
        "Flight",
        "Airline",
        "Seat",
        "Cabin",
        "ConfirmationCode",
        "Notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "Date": r.date_iso,
                    "From": r.origin_iata,
                    "To": r.destination_iata,
                    "Flight": r.flight_number,
                    "Airline": r.airline,
                    "Seat": r.seat,
                    "Cabin": r.cabin,
                    "ConfirmationCode": r.confirmation_code,
                    "Notes": r.notes,
                }
            )


overwrite_csv = write_csv


def _row_key(r: FlightRow) -> str:
    return f"{(r.flight_number or '').upper()}|{r.date_iso}"


def _first_non_empty(series: Iterable[Optional[str]]) -> str:
    for v in series:
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v
    return ""
def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@click.command()
@click.option("--start-date", type=str, default=MIN_DATE.strftime("%Y-%m-%d"), help="Filter flights on/after YYYY-MM-DD")
@click.option("--end-date", type=str, default=date.today().strftime("%Y-%m-%d"), help="Filter flights on/before YYYY-MM-DD")
@click.option("--output", type=click.Path(dir_okay=False, writable=True), default=DEFAULT_OUTPUT, help="Output CSV path")
@click.option("--user-name", type=str, default=os.environ.get("FLIGHTY_USER_NAME", "Neil"), help="Passenger name to match")
def main(start_date: str, end_date: str, output: str, user_name: str) -> None:
    start = _parse_date(start_date)
    end = _parse_date(end_date)

    # Build Gmail search window inclusive with a small buffer
    query = (
        f"after:{(start - timedelta(days=1)).strftime('%Y/%m/%d')} "
        f"before:{(end + timedelta(days=1)).strftime('%Y/%m/%d')} "
        + DEFAULT_QUERY
    )

    try:
        service = build_gmail_service()
        ids = search_message_ids(service, query)
        click.echo(f"[simple] Found {len(ids)} messages", err=False)

        output_path = os.path.abspath(output)

        thread_local = threading.local()

        def _thread_service() -> Any:
            svc = getattr(thread_local, "svc", None)
            if svc is None:
                # Build a separate Gmail client per thread
                svc = build_gmail_service()
                thread_local.svc = svc
            return svc

        def worker(mid: str) -> List[FlightRow]:
            svc = _thread_service()
            msg = get_message(svc, mid)
            text, html = extract_bodies(msg)
            rows = ai_extract_rows(text, html, user_name.strip())
            return filter_by_date(rows, start, end)

        processed = 0
        all_rows: List[FlightRow] = []
        with ThreadPoolExecutor(max_workers=30) as ex:
            futures = [ex.submit(worker, mid) for mid in ids]
            for fut in as_completed(futures):
                try:
                    batch = fut.result()
                except Exception as e:
                    processed += 1
                    click.echo(f"[simple] Error processing {processed}/{len(ids)}: {e}", err=True)
                    click.echo(traceback.format_exc(), err=True)
                    continue
                processed += 1
                click.echo(f"[simple] successfully processed {processed}/{len(ids)}", err=False)
                for r in batch:
                    click.echo(
                        f"[row] {r.date_iso} {r.origin_iata}->{r.destination_iata} "
                        f"{r.flight_number} {r.airline} seat={r.seat} cabin={r.cabin} "
                        f"conf={r.confirmation_code}",
                        err=False,
                    )
                all_rows.extend(batch)

        # Merge rows by key (Flight|Date), filling missing fields
        merged: Dict[str, FlightRow] = {}
        merges_count = 0
        for r in all_rows:
            key = _row_key(r)
            if key in merged:
                before = merged[key].__dict__.copy()
                _merge_rows(merged[key], r)
                if merged[key].__dict__ != before:
                    merges_count += 1
            else:
                merged[key] = r

        final_rows = sorted(merged.values(), key=lambda r: (r.date_iso, r.flight_number))
        overwrite_csv(final_rows, output_path)
        click.echo(f"[simple] Merged {merges_count} rows; wrote {len(final_rows)} total to: {output_path}")
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except HttpError as e:
        click.echo(f"Gmail API error: {e}", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()


