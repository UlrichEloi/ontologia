# Generated by Django 2.0.7 on 2018-08-07 23:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ontologia', '0006_fichier_etiquettes'),
    ]

    operations = [
        migrations.AddField(
            model_name='fichier',
            name='occurence_thematiques',
            field=models.TextField(default='', null=True),
        ),
        migrations.AddField(
            model_name='fichier',
            name='presence_thematiques',
            field=models.TextField(default='', null=True),
        ),
    ]
