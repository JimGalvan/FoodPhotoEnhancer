import base64
import logging
import uuid
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from enhancement_pipeline import EnhancementPipeline
from enhancer import get_subject_isolation_pipeline
from image_utils import ImageUtils

subject_sessions = {}
logger = logging.getLogger(__name__)


def index(request):
    return render(request, 'photo_enhancer/index.html')


@csrf_exempt
def upload_photo(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    if not request.FILES.get('photo'):
        return JsonResponse({'error': 'No photo file provided'}, status=400)

    try:
        photo = request.FILES['photo']
        image_bgr = ImageUtils.convert_payload_file_to_bgr(photo)

        subject_pipeline = get_subject_isolation_pipeline()
        subject_region, total_depth, _, _, image_source = subject_pipeline.find_subject(image_bgr)

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

        preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
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

        rgb_image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
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
