from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, username, password=None, is_active=True, is_staff=False, is_admin=False):
        if not username:
            raise ValueError("User must have an username")
        if not password:
            raise ValueError("User must have an password")
        user_obj = self.model(username=username)
        user_obj.set_password(password)
        user_obj.active = is_active
        user_obj.staff = is_staff
        user_obj.admin = is_admin
        user_obj.save(using=self._db)
        
        return user_obj

    def create_superuser(self,username, password=None, firstname=None):
        user = self.create_user(
            username,
            password=password,
            is_active=True,
            is_staff=True,
            is_admin=True,
        )

        return user

    def create_staffuser(self,username, password=None, firstname=None):
        user = self.create_user(
            username,
            password=password,
            is_active=True,
            is_staff=True,
        )

        return user



class User(AbstractBaseUser):
    # firstname       = models.CharField(max_length=30)
    # lastname        = models.CharField(max_length=30)
    Household_ID    = models.AutoField(primary_key=True)
    username           = models.CharField(unique=True, max_length=30)
    # contact_no      = models.CharField(max_length=13, null=True)
    # credit_source   = models.TextField(null=True)
    # salary          = models.DecimalField(decimal_places=2, max_digits=1000, null=True)
    # marital_status  = models.BooleanField(default=False)
    active          = models.BooleanField(default=True,null=True)
    staff           = models.BooleanField(default=False,null=True)
    admin           = models.BooleanField(default=False,null=True)

    USERNAME_FIELD = 'username'

    REQUIRED_FIELDS = []

    objects = UserManager()

    def __str__(self):
        return self.username

    def get_full_name(self):
        return self.username

    def get_short_name(self):
        return self.username

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True

    def get_email(self):
        return self.username

    @property
    def is_active(self):
        return self.active

    @property
    def is_staff(self):
        return self.staff

    @property
    def is_admin(self):
        return self.admin

    def set_password(self, raw_password):
        self.password = raw_password
        self._password = raw_password

    def check_password(self, raw_password):
        return self.password==raw_password


class UserData(models.Model):
    Household = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    First_Name = models.CharField(max_length=30, null=True)
    Last_Name = models.CharField(max_length=30, null=True)
    Gender = models.CharField(max_length=6, null=True)
    Age = models.IntegerField(null=True)
    Email = models.CharField(max_length=255,null=True)
    Contact_Number = models.CharField(max_length=20, null=True)
    Credit_Score = models.IntegerField(null=True)
    Tenure = models.IntegerField(null=True)
    Balance = models.FloatField(null=True)
    Any_Opened_Complaints = models.CharField(max_length=45, null=True)
    Marital_Status = models.CharField(max_length=45, null=True)
    Num_of_Insurance_Types = models.IntegerField(null=True)
    Is_Active = models.CharField(max_length=45, null=True)
    Estimated_Salary = models.FloatField(null=True)
    Churn = models.CharField(max_length=45, null=True)

    objects = models.Manager()
    
    def __str__(self):
        return self.First_Name




class ClaimData(models.Model):
    Row_ID = models.AutoField(primary_key=True)
    Household =  models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    Vehicle = models.IntegerField(null=True)
    Calendar_Year = models.IntegerField(null=True)
    Model_Year = models.IntegerField(null=True)
    Blind_Make = models.CharField(max_length=45, null=True)
    Blind_Model = models.CharField(max_length=45, null=True)
    Cat1 = models.CharField(max_length=45, null=True)
    Cat2 = models.CharField(max_length=45, null=True)
    Cat3 = models.CharField(max_length=45, null=True)
    Cat4 = models.CharField(max_length=45, null=True)
    Cat5 = models.CharField(max_length=45, null=True)
    Cat6 = models.CharField(max_length=45, null=True)
    Cat7 = models.CharField(max_length=45, null=True)
    Cat8 = models.CharField(max_length=45, null=True)
    Cat9 = models.CharField(max_length=45, null=True)
    Cat10 = models.CharField(max_length=45, null=True)
    Cat11 = models.CharField(max_length=45, null=True)
    Cat12 = models.CharField(max_length=45, null=True)
    OrdCat = models.CharField(max_length=45, null=True)
    Var1 = models.FloatField(null=True)
    Var2 = models.FloatField(null=True)
    Var3 = models.FloatField(null=True)
    Var4 = models.FloatField(null=True)
    Var5 = models.FloatField(null=True)
    Var6 = models.FloatField(null=True)
    Var7 = models.FloatField(null=True)
    Var8 = models.FloatField(null=True)
    Damage_Component = models.CharField(max_length=45, null=True)
    NVVar1 = models.FloatField(null=True)
    NVVar2 = models.FloatField(null=True)
    NVVar3 = models.FloatField(null=True)
    NVVar4 = models.FloatField(null=True)
    Claim_Amount = models.FloatField(null=True)
    Claim_Amount_Label = models.IntegerField(null=True)

    objects = models.Manager()

    def __str__(self):
        return self.Claim_Amount