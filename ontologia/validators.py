def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    #filename, ext = os.path.splitext(value)
    #valid_extensions = ['.pdf', '.doc', '.docx', '.jpg', '.png', '.xlsx', '.txt']
    valid_extensions = ['.pdf','.txt']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Extension de fichier non prise en charge. Entrer un fichier txt ou pdf')