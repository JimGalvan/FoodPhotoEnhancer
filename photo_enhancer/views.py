import base64
import os
import uuid
from io import BytesIO

import numpy as np
from PIL import Image
from django.shortcuts import render
from django.http import HttpResponse

from enhancement_pipeline import EnhancementPipeline
from config import settings
from enhancer import get_subject_isolation_pipeline


def index(request):
    return render(request, 'photo_enhancer/index.html')


def save_temp_photo(photo):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    ext = os.path.splitext(photo.name)[1]
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(temp_dir, filename)

    with open(path, 'wb+') as f:
        for chunk in photo.chunks():
            f.write(chunk)

    return path



def upload_photo(request):
    if request.method == 'POST' and request.FILES.get('photo'):
        photo = request.FILES['photo']
        path = save_temp_photo(photo)

        subject_pipeline = get_subject_isolation_pipeline()
        subject_region, total_depth, _, _, image_source = subject_pipeline.find_subject(path)

        enhance_pipeline = EnhancementPipeline(
            image=image_source,
            mask=subject_region.mask
        )
        final, clean_mask = enhance_pipeline.run()
        image = Image.fromarray(final.astype("uint8"))

        arr = final

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype("uint8")

        image = Image.fromarray(arr)

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return HttpResponse(
            f"""
            <p class="text-green-600 font-semibold mb-4">Photo processed successfully</p>
            <img src="data:image/jpeg;base64,{img_str}" class="w-full rounded shadow" />
            """
        )
