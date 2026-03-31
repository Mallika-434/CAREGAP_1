from django.apps import AppConfig


class PatientsConfig(AppConfig):
    name = 'patients'

    def ready(self):
        # Only run in the main process, not on every autoreload worker fork.
        # RUN_MAIN is set by Django's dev-server reloader; under gunicorn it
        # is absent, so we use an explicit opt-in env var for production too.
        import os
        if os.environ.get('RUN_MAIN') == 'true' or os.environ.get('WARM_CACHE_ON_START') == 'true':
            try:
                from django.core.management import call_command
                call_command('warm_cache')
            except Exception as e:
                print(f'warm_cache failed on startup: {e}')
