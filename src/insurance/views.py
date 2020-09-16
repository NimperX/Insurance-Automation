from django.shortcuts import redirect,render
from django.contrib import auth
from django.conf import settings
from django.contrib.auth import authenticate
from user.forms import CustomUserChangeForm, UserDataForm
from acc_claim.forms import AccidentClaimForm
from acc_claim.models import AccidentClaim
from user.models import UserData, ClaimData
from django.core.validators import ProhibitNullCharactersValidator
from os.path import join
from pathlib import Path
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from pickle import load as pickle_load
from pandas import DataFrame
from pycaret.classification import predict_model as classification_predict
from pycaret.classification import load_model as classification_load
from pycaret.regression import predict_model as regression_predict
from pycaret.regression import load_model as regression_load
from datetime import datetime
from torchvision import transforms, models
import torch
import torch.nn as nn
from skimage import io, transform
from PIL import Image



def home(request):
    if not request.user.is_authenticated:
        return redirect('login')

    return render(request, 'home.html')

def updateUser(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            form = CustomUserChangeForm(request.POST, instance=request.user)
            userdataform = UserDataForm(request.POST, instance=request.user.userdata)
            if form.is_valid() and userdataform.is_valid():
                form.save()
                userdataform.save()
                return redirect('/')
        
        user = request.user
        userdata = request.user.userdata
        form = CustomUserChangeForm(instance=user)
        userdataform = UserDataForm(instance=userdata)

        context = {
            'form': form,
            'userdataform': userdataform
        }
        return render(request, 'registration/update.html', context)

    return redirect('home')

def claimSuccess(request):
    churn_status = int(request.GET['churn'])

    if churn_status==0:
        context = {'churn_status': False}
    else:
        context = {'churn_status': True}

    return render(request, 'final.html', context)


def claimStatus(request):
    if request.user.is_authenticated:
        user_data = UserData.objects.filter(Household=request.user.Household_ID).first()
        claim_data = ClaimData.objects.filter(Household=request.user.Household_ID).latest('Row_ID')
        pred_component = {'H':'Head-light','R':'Rear mirror','T':'Tail-light','G':'Window','W': 'Windshield','D': 'Door'}

        churn_data = DataFrame(data={
            'Household_ID': [user_data.Household.Household_ID],
            'First_Name': [user_data.First_Name],
            'Last_Name': [user_data.Last_Name],
            'Gender': [user_data.Gender],
            'Age': [user_data.Age],
            'Email': [user_data.Email],
            'Contact_Number': [user_data.Contact_Number],
            'Credit_Score': [user_data.Credit_Score],
            'Tenure': [user_data.Tenure],
            'Balance': [user_data.Balance],
            'Any_Opened_Complaints': [user_data.Any_Opened_Complaints],
            'Marital_Status': [user_data.Marital_Status],
            'Num_of_Insurance_Types': [user_data.Num_of_Insurance_Types],
            'Is_Active': [user_data.Is_Active],
            'Estimated_Salary': [user_data.Estimated_Salary]
        })
        churn_pred_model = classification_load(join(Path(__file__).resolve(strict=True).parent,'models\\deployment_11092020'))
        churn_status = classification_predict(estimator=churn_pred_model, data=churn_data)['Label'].to_list()[0]

        context = {
            'claim_id': claim_data.Row_ID,
            'vehicle_model': ' '.join([claim_data.Blind_Make, claim_data.Blind_Model, str(claim_data.Model_Year)]),
            'damaged_comp': pred_component[claim_data.Damage_Component],
            'claim_amount': claim_data.Claim_Amount,
            'churn_status': True if churn_status=='Yes' else False
        }

        return render(request, 'claim_pred.html', context)


def accidentClaim(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            form = AccidentClaimForm(request.POST, request.FILES)

            if form.is_valid():
                form.save()
                data = AccidentClaim.objects.latest('id')
                img_path_damaged = data.damaged_image
                img_path_vehicle = data.vehicle_image

                # Vehicle classification model
                # vehiclemodel_model=load_model(join(Path(__file__).resolve(strict=True).parent,'models\\vehicle_makemodel.hdf5'))
                # img = image.load_img(join(settings.MEDIA_DIR,str(img_path_vehicle)), target_size=(299, 299))
                # x = image.img_to_array(img)
                # x = np.expand_dims(x, axis=0)
                # x = preprocess_input(x)

                # preds = vehiclemodel_model.predict(x)
                # pred_vehicle_model = np.argmax(preds)
                # pred_vehicle = {
                #     1:'Honda/Civic/2011/Car',
                #     2:'Toyota/Axio/2015/Car',
                #     3:'Toyota/Aqua/2019/Car',
                #     4:'Suzuki/Wragon R Stingray/2018/Car',
                #     5:'Audi/A8/2020/Car',
                #     6:'SsangYong/Actyon/2008/SUV',
                #     7:'Land Rover/Defender/2020/SUV',
                #     8:'Toyota/Land Cruiser Prado/2018/SUV',
                #     9:'Mitsubishi/Pajero/2018/SUV',
                #     10:'Range Rover/Autobiography/2020/SUV',
                #     11:'Honda/Vezel/2015/SUV',
                #     12:'Honda/Fit GP5/2015/Car',
                #     13:'Nissan/X-Trail Hybrid/2019/SUV',
                #     14:'Suzuki/Swift RS/2019/Car',
                #     15:'Toyota/Vitz/2019/Car'
                # }
                # print(pred_vehicle[pred_vehicle_model])

                batch_size = 4
                model_file_name = join(Path(__file__).resolve(strict=True).parent,'models\\vehicle_make_model_type_detection.pth')

                mean = [ 0.485, 0.456, 0.406 ]
                std = [ 0.229, 0.224, 0.225 ]

                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = mean, std = std),])

                test = []
                img_dir = join(settings.MEDIA_DIR,str(img_path_vehicle))
                label = -1
                test.append([img_dir, label])

                test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

                number_of_classes = 11
                model = models.resnet18(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, number_of_classes)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(model_file_name))
                else:
                    model.load_state_dict(torch.load(model_file_name, map_location=device))

                pred_vehicle_model = sample_tester(model, test_loader, transform = transform)
                pred_vehicle = {
                    'Audi A8 2020 Car': 'Audi/A8/2020/Car',
                    'Honda Civic 2011 Car': 'Honda/Civic/2011/Car',
                    'Honda Fit GP5 2015 Car': 'Honda/Fit GP5/2015/Car',
                    'Honda Vezel 2015 SUV': 'Honda/Vezel/2015/SUV',
                    'landcuiser predo 2018': 'Toyota/Land Cruiser Prado/2018/SUV',
                    'Nissan X -Trail Hybrid 2019 SUV': 'Nissan/X-Trail Hybrid/2019/SUV',
                    'Suzuki Wragon R Stingray 2018 Car': 'Suzuki/Wragon R Stingray/2018/Car',
                    'Suzuzki Swift RS 2019 Car': 'Suzuki/Swift RS/2019/Car',
                    'Toyota Aqua 2019 Car': 'Toyota/Aqua/2019/Car',
                    'Toyota Axio 2015 Car': 'Toyota/Axio/2015/Car',
                    'Toyota Vitz 2019 Car': 'Toyota/Vitz/2019/Car'
                }
                
                # Damage classification model
                damage_model=load_model(join(Path(__file__).resolve(strict=True).parent,'models\\model_resnet50_num.hdf5'))
                img = image.load_img(join(settings.MEDIA_DIR,str(img_path_damaged)), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = damage_model.predict(x)
                pred_name = np.argmax(preds)
                pred_component = {0:'Head-light',1:'Rear mirror',2:'Tail-light',3:'Window',4: 'Windshield',5: 'Door'}
                pred_comp_for_pd = {0: 'H', 1: 'R', 2: 'T', 3: 'G', 4: 'W', 5: 'D'}
                print(pred_component[pred_name])
                #Must Add Dictionary {'Head-light':0,'Rear mirror':1,'Tail-light':2,'Window':3, 'Windshield':4, 'Door':5}

                #Model for claim classification
                claim_pred_model_classi = classification_load(join(Path(__file__).resolve(strict=True).parent,'models\\Pycaret2.1.1_Final_Blend1_Classification'))

                user_data = UserData.objects.filter(Household=request.user.Household_ID).first()
                claim_data = ClaimData.objects.filter(Household=request.user.Household_ID).first()

                predict_data = DataFrame(data={
                    'Row_ID':[ClaimData.objects.latest('Row_ID').Row_ID + 1],
                    'Household_ID': [claim_data.Household],
                    'Vehicle': [len(ClaimData.objects.filter(Household=request.user.Household_ID)) + 1],
                    'Calendar_Year': [datetime.now().year],
                    'Model_Year': [pred_vehicle[pred_vehicle_model].split('/')[2]],
                    'Blind_Make': [pred_vehicle[pred_vehicle_model].split('/')[0]],
                    'Blind_Model': [pred_vehicle[pred_vehicle_model].split('/')[1]],
                    'Cat1': [claim_data.Cat1],
                    'Cat2': [claim_data.Cat2],
                    'Cat3': [claim_data.Cat3],
                    'Cat4': [claim_data.Cat4],
                    'Cat5': [claim_data.Cat5],
                    'Cat6': [claim_data.Cat6],
                    'Cat7': [claim_data.Cat7],
                    'Cat8': [claim_data.Cat8],
                    'Cat9': [claim_data.Cat9],
                    'Cat10': [claim_data.Cat10],
                    'Cat11': [claim_data.Cat11],
                    'Cat12': [claim_data.Cat12],
                    'OrdCat': [claim_data.OrdCat],
                    'Var1': [claim_data.Var1],
                    'Var2': [claim_data.Var2],
                    'Var3': [claim_data.Var3],
                    'Var4': [claim_data.Var4],
                    'Var5': [claim_data.Var5],
                    'Var6': [claim_data.Var6],
                    'Var7': [claim_data.Var7],
                    'Var8': [claim_data.Var8],
                    'Damage_Component': [pred_comp_for_pd[pred_name]],
                    'NVVar1': [claim_data.NVVar1],
                    'NVVar2': [claim_data.NVVar2],
                    'NVVar3': [claim_data.NVVar3],
                    'NVVar4': [claim_data.NVVar4],
                })
                claim_status = classification_predict(estimator=claim_pred_model_classi, data=predict_data)['Label'].to_list()[0]
                claim_status = 1 #Remove this line to remove override
                
                if claim_status==1:
                    #Model for claim regression
                    claim_pred_model_reg = regression_load(join(Path(__file__).resolve(strict=True).parent,'models\\Pycaret2.1.2_Regression_Final'))
                    predict_data.insert(33,'Claim_Amount_Label',[1],True)
                    claim_amount = regression_predict(estimator=claim_pred_model_reg, data=predict_data)['Label'].to_list()[0]
                    claim_amount = float("{:.2f}".format(claim_amount))

                churn_data = DataFrame(data={
                    'Household_ID': [user_data.Household.Household_ID],
                    'First_Name': [user_data.First_Name],
                    'Last_Name': [user_data.Last_Name],
                    'Gender': [user_data.Gender],
                    'Age': [user_data.Age],
                    'Email': [user_data.Email],
                    'Contact_Number': [user_data.Contact_Number],
                    'Credit_Score': [user_data.Credit_Score],
                    'Tenure': [user_data.Tenure],
                    'Balance': [user_data.Balance],
                    'Any_Opened_Complaints': [user_data.Any_Opened_Complaints],
                    'Marital_Status': [user_data.Marital_Status],
                    'Num_of_Insurance_Types': [user_data.Num_of_Insurance_Types],
                    'Is_Active': [user_data.Is_Active],
                    'Estimated_Salary': [user_data.Estimated_Salary]
                })
                churn_pred_model = classification_load(join(Path(__file__).resolve(strict=True).parent,'models\\deployment_11092020'))
                churn_status = classification_predict(estimator=churn_pred_model, data=churn_data)['Label'].to_list()[0]

                newClaimData = ClaimData(
                    Household = claim_data.Household,
                    Vehicle = len(ClaimData.objects.filter(Household=request.user.Household_ID)) + 1,
                    Calendar_Year = datetime.now().year,
                    Model_Year = pred_vehicle[pred_vehicle_model].split('/')[2],
                    Blind_Make = pred_vehicle[pred_vehicle_model].split('/')[0],
                    Blind_Model = pred_vehicle[pred_vehicle_model].split('/')[1],
                    Cat1 = claim_data.Cat1,
                    Cat2 = claim_data.Cat2,
                    Cat3 = claim_data.Cat3,
                    Cat4 = claim_data.Cat4,
                    Cat5 = claim_data.Cat5,
                    Cat6 = claim_data.Cat6,
                    Cat7 = claim_data.Cat7,
                    Cat8 = claim_data.Cat8,
                    Cat9 = claim_data.Cat9,
                    Cat10 = claim_data.Cat10,
                    Cat11 = claim_data.Cat11,
                    Cat12 = claim_data.Cat12,
                    OrdCat = claim_data.OrdCat,
                    Var1 = claim_data.Var1,
                    Var2 = claim_data.Var2,
                    Var3 = claim_data.Var3,
                    Var4 = claim_data.Var4,
                    Var5 = claim_data.Var5,
                    Var6 = claim_data.Var6,
                    Var7 = claim_data.Var7,
                    Var8 = claim_data.Var8,
                    Damage_Component = pred_comp_for_pd[pred_name],
                    NVVar1 = claim_data.NVVar1,
                    NVVar2 = claim_data.NVVar2,
                    NVVar3 = claim_data.NVVar3,
                    NVVar4 = claim_data.NVVar4,
                    Claim_Amount = claim_amount,
                    Claim_Amount_Label = claim_status
                )
                newClaimData.save()

                context = {
                    'claim_id': ClaimData.objects.latest('Row_ID').Row_ID,
                    'vehicle_model': ' '.join(pred_vehicle[pred_vehicle_model].split('/')[:-1]),
                    'damaged_comp': pred_component[pred_name],
                    'claim_amount': claim_amount,
                    'churn_status': True if churn_status=='Yes' else False
                }

                return render(request, 'claim_pred.html', context)

        form = AccidentClaimForm()
        context = {
            'form': form
        }
        return render(request, 'acc_claim.html', context)



def read_load_inputs_labels(image_names, labels, transform = None):
	img_inputs = []
	for i in range(len(image_names)):
		image = io.imread(image_names[i])
		if transform:
			image = transform(Image.fromarray(image))
		img_inputs.append(image)
	inputs = torch.stack(img_inputs)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	inputs = inputs.to(device)
	labels = labels.to(device)
	return inputs, labels


def sample_tester(model, test_loader, transform = None):
	model.eval()
	label_name_list = ['Audi A8 2020 Car', 'Honda Civic 2011 Car', 'Honda Fit GP5 2015 Car', 'Honda Vezel 2015 SUV', 'landcuiser predo 2018', 'Nissan X -Trail Hybrid 2019 SUV', 'Suzuki Wragon R Stingray 2018 Car', 'Suzuzki Swift RS 2019 Car', 'Toyota Aqua 2019 Car', 'Toyota Axio 2015 Car', 'Toyota Vitz 2019 Car']
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for k in range(preds.size(0)):
				return label_name_list[preds[k]]