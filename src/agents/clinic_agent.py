"""Structured clinic information and JSON-backed appointment operations."""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid
from typing import Any
from src.llm.client import LLMClient
from pathlib import Path

from src.config import AppSettings, get_settings
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class ClinicAgent:
    """Read clinic facts from Qdrant and persist bookings in Qdrant state."""

    _lock = threading.RLock()

    def __init__(self, clinic_path: Path | None = None, bookings_path: Path | None = None, settings: AppSettings | None = None, booking_store: Any | None = None, llm_client: LLMClient | None = None) -> None:
        app_settings = settings or get_settings()
        self.settings = app_settings
        self.llm_client = llm_client or LLMClient(default_model=app_settings.llm.generator_model, settings=app_settings)
        self.booking_store = booking_store
        if self.booking_store is None and bookings_path is None and app_settings.runtime.state_backend == "qdrant":
            from src.state import QdrantStateStore

            try:
                self.booking_store = QdrantStateStore.from_settings(app_settings)
            except RuntimeError:
                self.booking_store = None
        self.clinic_path = clinic_path or app_settings.resolve_path(app_settings.agents.clinic_data_path)
        self.bookings_path = bookings_path or app_settings.resolve_path(app_settings.agents.bookings_path)
        if self.clinic_path.exists():
            self.clinic = json.loads(self.clinic_path.read_text(encoding="utf-8"))
        elif app_settings.runtime.state_backend == "qdrant":
            self.clinic = self._load_clinic_kb()
        else:
            self.clinic = self._empty_clinic()

    @staticmethod
    def _empty_clinic() -> dict[str, Any]:
        """Return an empty catalog when no source collection is configured."""
        return {"clinic_info": {}, "doctors": [], "services": [], "appointments_policy": {}}

    def _load_clinic_kb(self) -> dict[str, Any]:
        """Load the deployed clinic catalog from Qdrant chunks, not disk."""
        try:
            client = getattr(self.booking_store, "client", None)
            if client is None:
                return self._empty_clinic()
            records, _ = client.scroll(
                collection_name=self.settings.vectordb.clinic_collection,
                limit=128,
                with_payload=True,
                with_vectors=False,
            )
            texts = [str((point.payload or {}).get("text", "")) for point in records]
            return {"clinic_info": {}, "doctors": [], "services": [], "appointments_policy": {}, "_chunks": texts} if texts else self._empty_clinic()
        except Exception:
            return self._empty_clinic()

    @staticmethod
    def _norm(value: object) -> str:
        return str(value or "").casefold().translate(_ARABIC_DIGITS).strip()

    def _doctors(self, entities: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        entities = entities or {}
        doctor_id = str(entities.get("doctor_id") or "").strip()
        doctors = self.clinic.get("doctors", [])
        if doctor_id:
            matched = [d for d in doctors if d.get("id") == doctor_id]
            if matched:
                return matched
        terms = [self._norm(entities.get(key)) for key in ("doctor", "specialty")]
        terms = [term for term in terms if term]
        if not terms:
            return doctors
        matched = []
        for doctor in doctors:
            haystack = self._norm(" ".join(str(doctor.get(key, "")) for key in ("name", "name_en", "specialization", "specialization_en", "id")))
            if all(term in haystack for term in terms):
                matched.append(doctor)
        return matched

    def handle(self, intent: str = "clinic_info", entities: dict[str, Any] | None = None, *, user_ref: str = "anonymous", query: str = "", **_: object) -> dict[str, Any]:
        if intent == "booking":
            return self.book(entities or {}, user_ref=user_ref, query=query)
        if intent == "cancellation":
            return self.cancel(str((entities or {}).get("booking_id") or ""), user_ref=user_ref)
        if intent == "confirmation":
            return self.confirm(str((entities or {}).get("booking_id") or ""), user_ref=user_ref)
        return self.lookup(intent=intent, entities=entities or {}, query=query)

    def lookup(self, *, intent: str = "clinic_info", entities: dict[str, Any] | None = None, query: str = "") -> dict[str, Any]:
        entities = entities or {}
        # Use the LLM to naturally and accurately understand and answer user queries
        if query and query.strip() and self.llm_client is not None:
            answer = self._llm_lookup(intent, entities, query)
            return {"route": "clinic_query", "answer": answer, "data": {"intent": intent}, "citations": []}

        # Offline / deterministic fallback when no LLM or query is empty (e.g. unit tests)
        doctors = self._doctors(entities)
        if intent == "availability":
            day = self._weekday(entities)
            if not doctors:
                return {"route": "clinic_query", "answer": "لا يوجد طبيب مطابق لهذا التخصص في بيانات العيادة حالياً.", "data": {"doctors": []}, "citations": []}
            rows = [{"doctor": d.get("name"), "specialization": d.get("specialization"), "day": day, "slots": d.get("schedule", {}).get(day, []) if day else d.get("schedule", {})} for d in doctors]
            return {"route": "clinic_query", "answer": self._format_availability(rows, day), "data": {"doctors": rows}, "citations": []}

        return {"route": "clinic_query", "answer": self._format_info(), "data": self.clinic, "citations": []}

    def _format_clinic_context(self) -> str:
        """Format full clinic catalog into clear reference text for the LLM."""
        if self.clinic.get("_chunks"):
            return "\n\n---\n\n".join(self.clinic["_chunks"])

        parts = []
        info = self.clinic.get("clinic_info", {})
        if info:
            parts.append(
                f"معلومات العيادة:\nالاسم: {info.get('name', '')} ({info.get('name_en', '')})\n"
                f"العنوان: {info.get('address', '')}\nالهاتف: {info.get('phone', '')}\n"
                f"طوارئ: {info.get('emergency_phone', '')}"
            )
        hours = self.clinic.get("working_hours", {})
        if hours:
            h_lines = [
                f"- {day}: {h.get('open', '')}-{h.get('close', '')} ({h.get('note', '')})"
                for day, h in hours.items() if isinstance(h, dict)
            ]
            parts.append("ساعات العمل العامة:\n" + "\n".join(h_lines))
        doctors = self.clinic.get("doctors", [])
        if doctors:
            d_lines = []
            for d in doctors:
                sched = d.get("schedule", {})
                s_str = ", ".join(f"{day}: {', '.join(slots)}" for day, slots in sched.items() if slots)
                d_lines.append(
                    f"الطبيب: {d.get('name', '')} ({d.get('name_en', '')})\n"
                    f"الكود: {d.get('id', '')}\n"
                    f"التخصص: {d.get('specialization', '')} ({d.get('specialization_en', '')})\n"
                    f"سعر الكشف: {d.get('consultation_fee', '')} {d.get('currency', 'EGP')}\n"
                    f"المواعيد المتاحة:\n{s_str}\n"
                    f"ملاحظات: {d.get('notes', '')}"
                )
            parts.append("قائمة الأطباء وجداول المواعيد:\n\n" + "\n\n".join(d_lines))
        services = self.clinic.get("services", [])
        if services:
            s_lines = [
                f"- {s.get('name', '')}: {s.get('price', '')} {s.get('currency', 'EGP')} ({s.get('category', '')})"
                for s in services
            ]
            parts.append("الخدمات والأسعار:\n" + "\n".join(s_lines))
        policy = self.clinic.get("appointments_policy", {})
        if policy:
            parts.append(
                f"سياسات المواعيد:\nطرق الحجز: {', '.join(policy.get('booking_methods', []))}\n"
                f"سياسة الإلغاء: {policy.get('cancellation_policy', '')}"
            )
        return "\n\n====================\n\n".join(parts)

    def _llm_lookup(self, intent: str, entities: dict[str, Any], query: str) -> str:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / "clinic_answer.txt"
        system = prompt_path.read_text(encoding="utf-8")
        data = self._format_clinic_context()
        user = f"USER QUESTION: {query}\nINTENT: {intent}\nDATA:\n{data}"
        try:
            response = self.llm_client.complete(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=self.settings.llm.generator_model,
                temperature=0,
                num_retries=1,
            )
            text = str(response.choices[0].message.content or "").strip()
            return text if text else "لا توجد معلومات كافية في سجلات العيادة لهذا السؤال."
        except Exception:
            return "لا أستطيع الوصول إلى معلومات العيادة الآن. حاول مرة أخرى لاحقًا."

    def _format_info(self) -> str:
        if self.clinic.get("_chunks"):
            return "\n\n".join(self.clinic["_chunks"])
        info = self.clinic.get("clinic_info", {})
        policy = self.clinic.get("appointments_policy", {})
        services = self.clinic.get("services", [])
        prices = "\n".join(
            f"- {service.get('name')}: {service.get('price')} {service.get('currency', 'EGP')}"
            for service in services
        )
        return (
            f"{info.get('name', '')}\nالعنوان: {info.get('address', '')}\n"
            f"الهاتف: {info.get('phone', '')}\n"
            f"الأسعار والخدمات:\n{prices}\n"
            f"طرق الحجز: {', '.join(policy.get('booking_methods', []))}\n"
            f"سياسة الإلغاء: {policy.get('cancellation_policy', '')}"
        )

    @staticmethod
    def _format_availability(rows: list[dict[str, Any]], day: str | None) -> str:
        output = []
        for row in rows:
            slots = row["slots"]
            if isinstance(slots, dict):
                slots = ", ".join(f"{key}: {', '.join(value)}" for key, value in slots.items() if value)
            else:
                slots = ", ".join(slots) if slots else "لا توجد مواعيد مسجلة"
            output.append(f"{row['doctor']} ({row['specialization']}): {slots}")
        prefix = f"مواعيد {day}:\n" if day else "المواعيد المتاحة:\n"
        return prefix + "\n".join(output)

    @staticmethod
    def _weekday(entities: dict[str, Any]) -> str | None:
        value = str(entities.get("weekday") or "").casefold()
        aliases = {"السبت": "saturday", "الأحد": "sunday", "الاحد": "sunday", "الاثنين": "monday", "الإثنين": "monday", "الثلاثاء": "tuesday", "الأربعاء": "wednesday", "الخميس": "thursday", "الجمعة": "friday"}
        if value in aliases:
            return aliases[value]
        return value if value in _WEEKDAYS else None

    def _read_bookings(self) -> list[dict[str, Any]]:
        if self.booking_store is not None:
            return self.booking_store.list("booking")
        if not self.bookings_path.exists():
            return []
        payload = json.loads(self.bookings_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("bookings", [])

    def _write_bookings(self, bookings: list[dict[str, Any]]) -> None:
        if self.booking_store is not None:
            for booking in bookings:
                booking_id = str(booking.get("booking_id") or "")
                if booking_id:
                    self.booking_store.put("booking", booking_id, booking)
            return
        self.bookings_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".bookings-", suffix=".json", dir=self.bookings_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(bookings, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.bookings_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _find_doctor(self, entities: dict[str, Any], query: str = "") -> dict[str, Any] | None:
        doctor_id = str(entities.get("doctor_id") or "").strip()
        doctors = self.clinic.get("doctors", [])
        if doctor_id:
            for doctor in doctors:
                if doctor.get("id") == doctor_id:
                    return doctor
        matched = self._doctors(entities)
        if len(matched) == 1:
            return matched[0]

        if (query or entities) and self.llm_client is not None:
            doctor = self._find_doctor_with_llm(entities, query)
            if doctor:
                return doctor

        return matched[0] if len(matched) == 1 else None

    def _find_doctor_with_llm(self, entities: dict[str, Any], query: str = "") -> dict[str, Any] | None:
        if self.llm_client is None:
            return None
        doctors = self.clinic.get("doctors", [])
        if not doctors:
            return None
        doctors_summary = "\n".join(
            f"- ID: {d.get('id')}, Name: {d.get('name')}, Specialty: {d.get('specialization')} ({d.get('specialization_en')})"
            for d in doctors
        )
        prompt = (
            "Given this list of clinic doctors:\n"
            f"{doctors_summary}\n\n"
            f"User request: {query}\n"
            f"Entities: {entities}\n\n"
            "Which doctor ID does the user want to book with? "
            "Return ONLY the exact doctor ID (e.g. DR001) without any other words, or 'NONE' if no doctor or specialty is specified."
        )
        try:
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings.llm.classifier_model,
                temperature=0,
                num_retries=1,
            )
            raw = str(response.choices[0].message.content or "").strip()
            for d in doctors:
                if d.get("id") == raw or (len(raw) <= 10 and d.get("id") in raw):
                    return d
        except Exception:
            return None
        return None

    def book(self, entities: dict[str, Any], *, user_ref: str, query: str = "") -> dict[str, Any]:
        doctor = self._find_doctor(entities, query)
        appointment_date = str(entities.get("date") or "")
        start_time = str(entities.get("time") or "")
        missing = []
        if doctor is None:
            missing.append("doctor or specialty")
        if not appointment_date:
            missing.append("date")
        if not start_time:
            missing.append("time")
        if missing:
            return {"route": "clinic_query", "answer": "لحجز الموعد أحتاج: " + ", ".join(missing) + ".", "status": "missing_fields", "missing_fields": missing, "citations": []}
        try:
            parsed_date = date.fromisoformat(appointment_date)
        except ValueError:
            return {"route": "clinic_query", "answer": "صيغة التاريخ يجب أن تكون YYYY-MM-DD.", "status": "invalid", "citations": []}
        weekday = _WEEKDAYS[parsed_date.weekday()]
        slots = doctor.get("schedule", {}).get(weekday, [])
        if not self._time_in_slots(start_time, slots):
            return {"route": "clinic_query", "answer": "هذا الوقت غير موجود ضمن جدول الطبيب في ذلك اليوم.", "status": "unavailable", "citations": []}
        with self._lock:
            bookings = self._read_bookings()
            conflict = any(b.get("status") in {"pending", "confirmed"} and b.get("doctor_id") == doctor["id"] and b.get("appointment_date") == appointment_date and b.get("start_time") == start_time for b in bookings)
            if conflict:
                return {"route": "clinic_query", "answer": "هذا الموعد محجوز بالفعل.", "status": "conflict", "citations": []}
            now = datetime.now().isoformat(timespec="seconds")
            booking = {"booking_id": "BK-" + uuid.uuid4().hex[:10].upper(), "user_ref": user_ref, "doctor_id": doctor["id"], "doctor_name": doctor.get("name"), "appointment_date": appointment_date, "start_time": start_time, "status": "pending", "created_at": now, "updated_at": now}
            bookings.append(booking)
            self._write_bookings(bookings)
        return {"route": "clinic_query", "answer": f"تم إنشاء طلب الحجز {booking['booking_id']} مع {doctor.get('name')} يوم {appointment_date} الساعة {start_time}. أرسل تأكيداً لإتمام الحجز.", "status": "pending", "booking": booking, "citations": []}

    @staticmethod
    def _time_in_slots(start_time: str, slots: list[str]) -> bool:
        try:
            target = datetime.strptime(start_time, "%H:%M").time()
        except ValueError:
            return False
        for slot in slots:
            begin, end = slot.split("-", 1)
            if datetime.strptime(begin, "%H:%M").time() <= target < datetime.strptime(end, "%H:%M").time():
                return True
        return False

    def _mutate_status(self, booking_id: str, user_ref: str, status: str) -> dict[str, Any]:
        with self._lock:
            bookings = self._read_bookings()
            for booking in bookings:
                if booking.get("booking_id") == booking_id and booking.get("user_ref") == user_ref:
                    if status == "confirmed" and booking.get("status") != "pending":
                        return {"route": "clinic_query", "answer": "هذا الطلب لا يمكن تأكيده حالياً.", "status": "invalid", "citations": []}
                    if status == "cancelled" and booking.get("status") not in {"pending", "confirmed"}:
                        return {"route": "clinic_query", "answer": "هذا الحجز ملغى بالفعل.", "status": "invalid", "citations": []}
                    booking["status"] = status
                    booking["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    self._write_bookings(bookings)
                    verb = "تم تأكيد" if status == "confirmed" else "تم إلغاء"
                    return {"route": "clinic_query", "answer": f"{verb} الحجز {booking_id}.", "status": status, "booking": booking, "citations": []}
        return {"route": "clinic_query", "answer": "لم يتم العثور على حجز مطابق.", "status": "not_found", "citations": []}

    def confirm(self, booking_id: str, *, user_ref: str) -> dict[str, Any]:
        return self._mutate_status(booking_id, user_ref, "confirmed")

    def cancel(self, booking_id: str, *, user_ref: str) -> dict[str, Any]:
        return self._mutate_status(booking_id, user_ref, "cancelled")
