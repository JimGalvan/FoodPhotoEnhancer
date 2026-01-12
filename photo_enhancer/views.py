import base64
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from enhancement_pipeline import EnhancementPipeline
from config import settings
from enhancer import get_subject_isolation_pipeline
import os
import uuid

subject_sessions = {}
logger = logging.getLogger(__name__)


def index(request):
    return render(request, 'photo_enhancer/index.html')


def save_temp_photo(photo):
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    ext = os.path.splitext(photo.name)[1]
    final_name = f"{uuid.uuid4()}{ext}"
    final_path = os.path.join(temp_dir, final_name)
    tmp_path = final_path + ".tmp"

    logger.info(
        "Starting image write",
        extra={
            "original_name": photo.name,
            "tmp_path": tmp_path,
            "final_path": final_path,
        },
    )

    total_bytes = 0

    try:
        with open(tmp_path, "wb") as f:
            for i, chunk in enumerate(photo.chunks()):
                chunk_size = len(chunk)
                f.write(chunk)
                total_bytes += chunk_size

                logger.debug(
                    "Wrote image chunk",
                    extra={
                        "chunk_index": i,
                        "chunk_size": chunk_size,
                        "total_bytes": total_bytes,
                        "tmp_path": tmp_path,
                    },
                )

            f.flush()
            os.fsync(f.fileno())

        logger.info(
            "Finished writing temp image",
            extra={
                "tmp_path": tmp_path,
                "total_bytes": total_bytes,
            },
        )

        os.rename(tmp_path, final_path)

        logger.info(
            "Image write committed (atomic rename)",
            extra={
                "final_path": final_path,
                "total_bytes": total_bytes,
            },
        )

        return final_path

    except Exception as e:
        logger.exception(
            "Failed while saving temp photo",
            extra={
                "tmp_path": tmp_path,
                "final_path": final_path,
                "bytes_written": total_bytes,
            },
        )

        # Cleanup partial file if it exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.warning(
                    "Removed partial temp file after failure",
                    extra={"tmp_path": tmp_path},
                )
        except Exception:
            logger.exception("Failed to clean up temp file")

        raise


def delete_file(path: str) -> None:
    p = Path(path)
    if p.exists() and p.is_file():
        p.unlink()


@csrf_exempt
def upload_photo(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    if not request.FILES.get('photo'):
        return JsonResponse({'error': 'No photo file provided'}, status=400)

    try:
        photo = request.FILES['photo']
        path = save_temp_photo(photo)

        subject_pipeline = get_subject_isolation_pipeline()
        subject_region, total_depth, _, _, image_source = subject_pipeline.find_subject(path)

        delete_file(path)

        session_id = str(uuid.uuid4())
        subject_sessions[session_id] = {
            'subject_region': subject_region,
            'total_depth': total_depth,
            'image_source': image_source
        }

        preview_image = image_source.copy()
        mask = (subject_region.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview_image, contours, -1, (0, 255, 0), 3)

        if preview_image.dtype != np.uint8:
            preview_image = np.clip(preview_image, 0, 255).astype(np.uint8)

        image = Image.fromarray(preview_image)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return JsonResponse({
            'success': True,
            'message': 'Subject detected successfully',
            'session_id': session_id,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def enhance_photo(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    session_id = request.POST.get('session_id')
    if not session_id or session_id not in subject_sessions:
        return JsonResponse({'error': 'Invalid or expired session'}, status=400)

    try:
        session_data = subject_sessions.pop(session_id)
        subject_region = session_data['subject_region']
        image_source = session_data['image_source']

        enhance_pipeline = EnhancementPipeline(
            image=image_source,
            mask=subject_region.mask
        )
        final, clean_mask = enhance_pipeline.run()

        arr = final
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype("uint8")

        image = Image.fromarray(arr)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return JsonResponse({
            'success': True,
            'message': 'Photo enhanced successfully',
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
