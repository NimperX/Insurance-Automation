from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin

from .forms import CustomUserCreationForm, CustomUserChangeForm
from .models import User
from django.utils.translation import gettext, gettext_lazy as _

class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = User
    list_display = ('username',)
    list_filter = ()  # Contain only fields in your `custom-user-model` intended for filtering. Do not include `groups`since you do not have it
    search_fields = ()  # Contain only fields in your `custom-user-model` intended for searching
    ordering = ()  # Contain only fields in your `custom-user-model` intended to ordering
    filter_horizontal = () # Leave it empty. You have neither `groups` or `user_permissions`
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        # (_('Personal info'), {'fields': ('firstname', 'lastname','contact_no','credit_source','salary','marital_status',)}),
        (_('Permissions'), {
            'fields': ('active', 'staff', 'admin'),
        }),
    )

admin.site.register(User, CustomUserAdmin)