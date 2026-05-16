from django.conf import settings


def cache_bust(request):
    return {'CACHE_BUST': getattr(settings, 'CACHE_BUST', '1')}
