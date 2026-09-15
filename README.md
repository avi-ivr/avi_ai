# Gemini IVR

## Render

- Build Command: `python -m pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30`
- Health URL: `/health`
- Refresh URL: `/refresh`
- Demo URL: `/demo`

אפשר לקרוא ל-`/refresh` כל 10 דקות משירות תזמון חיצוני. Render עצמו אינו מבטיח שהתעוררות דרך health check תשאיר שירות חינמי ער.

לפני ההפעלה יש להוסיף מפתחות אמיתיים לקובץ `Api_Kay_gemini.txt`, מפתח בכל שורה. המערכת מנסה אותם לפי הסדר ועוברת לבא במקרה של כשל.

יש להגדיר ב-Render את `WEBHOOK_SECRET`, `PUBLIC_WEBHOOK_SECRET` ו-`TRAINER_WEBHOOK_SECRET`.

לאחר קבלת כתובת Render, יש לעדכן אותה ב-`ext.ini` וב-`trainer_ext.ini` בשדה `api_link`.

חשוב: אם Render מוגדר ידנית ולא משתמש ב-`render.yaml`, יש להגדיר את ה-Build Command בדיוק כך:

```text
python -m pip install -r requirements.txt
```

התמלול מתבצע באמצעות Gemini מקובץ השמע שהורד מימות. אין הורדה או טעינה של מודל Whisper ב-Render.

## פריסה מלאה ב-Render

1. העלה ל-GitHub את כל תוכן התיקייה הזו.
2. ב-Render צור `New +` ואז `Web Service` וחבר את המאגר.
3. הגדר:
	- Root Directory: השאר ריק אם הקבצים בשורש המאגר.
	- Build Command: `python -m pip install -r requirements.txt`
	- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30`
4. הוסף משתני סביבה:
	- `GEMINI_MODEL=gemini-3.6-flash`
	- `WEBHOOK_SECRET`
	- `PUBLIC_WEBHOOK_SECRET`
	- `TRAINER_WEBHOOK_SECRET`
	- `GEMINI_TRANSCRIBE_MODEL=gemini-3.5-transcribe`
5. בצע Deploy וחכה שיופיע `Gemini transcription build completed successfully` בלוג ה-Build.
6. בדוק בדפדפן:
	- `https://avi-ai.onrender.com/health`
	- `https://avi-ai.onrender.com/debug`
7. בימות הגדר ב-`ext.ini`:
	- `api_link=https://avi-ai.onrender.com/yemot`
	- `api_add_0=app_role=public`
	- `api_add_1=yemot_token=TOKEN של ימות`

## בדיקת שגיאה

בכל בקשת ימות יופיע בלוג:

```text
[YEMOT REQUEST] {...}
```

אם הורדת או תמלול ההקלטה נכשלו, חפש בלוג את:

```text
Failed to download or transcribe Yemot recording
```

השורה שאחריו כוללת את סוג השגיאה, שם הקובץ והתיקייה.
