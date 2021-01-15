# importing the necessary dependencies
from flask import Flask, render_template, request,send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
matplotlib.use('Agg')

main = Flask(__name__) # initializing a flask app
# app=application


@main.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@main.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            col_name=str(request.form['State_name'])
            Year = int(request.form['Year'])
            Day = int(request.form['Day'])
            Month = int(request.form['Month'])                    # loading the model file from the storage

            from datetime import datetime
            from datetime import timedelta
            from pandas import Series
            from datetime import date
            import pandas as pd
            import numpy as np

            def difference(dataset, interval=1):
                diff = list()
                for i in range(interval, len(dataset)):
                    value = dataset[i] - dataset[i - interval]
                    diff.append(value)
                return Series(diff)

                differenced = difference(new_df, 1)
                differenced.head()

            d0 = date(Year, Month, Day)
            d1 = date(2020, 11, 14)
            delta = d0 - d1
            no_days = delta.days
            from keras.models import load_model
            filename1 =  'models_with_tuner/'+col_name+'_model_100.h5'

            model = load_model( filepath=filename1 )
            model.load_weights( filepath=filename1 )

            if col_name in ['telangana', 'andhra pradesh']:

                data1 = pd.read_csv('pycharm_data.csv')
                data = list(data1[col_name][-39:])
                differenced = difference(data, 1)
                X_test = pd.DataFrame(differenced)
                x_input = np.array(X_test)
                lst_output = []
                i = 0

                while (i < no_days):
                    if (len(data) > 39):
                        X_test = pd.DataFrame(difference(data[i:], 1))
                        x_input = np.array(X_test)
                        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
                        x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
                        yhat = model.predict(x_input, verbose=0)
                        yhat = yhat + data[-1]
                        lst_output.append(yhat[0][0])
                        data.append(yhat[0][0])
                        i = i + 1

                    else:
                        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
                        x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
                        yhat = model.predict(x_input, verbose=0)
                        yhat = yhat + data[-1]
                        lst_output.append(yhat[0][0])
                        data.append(yhat[0][0])
                        i = i + 1
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                from io import StringIO
                import base64
                img = io.BytesIO()
                sns.set_style("dark")  # E.G.

                y = lst_output
                x = [x for x in range(len(lst_output))]

                plt.plot(x, y)
                plt.xlabel('Day count after 14/11/2020')
                plt.ylabel('MegaUnits')
                plt.savefig(img, format='png')
                plt.close()
                img.seek(0)

                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                print(plot_url)

                #
                #
                #
                #
                # gr = sns.lineplot(x=[x for x in range(len(lst_output))], y=lst_output)
                # import os
                # strFile = "static/image/pred.jpg"
                # if os.path.exists(strFile):
                #     os.remove(strFile)  # Opt.: os.system("rm "+strFile)
                # gr.figure.savefig(strFile)

            else:

                data1 = pd.read_csv('pycharm_data.csv')
                data = list(data1[col_name][-49:])
                differenced = difference(data, 1)
                X_test = pd.DataFrame(differenced)
                x_input = np.array(X_test)
                lst_output = []
                i = 0
                while (i < no_days):
                    if (len(data) > 49):
                        X_test = pd.DataFrame(difference(data[i:], 1))
                        x_input = np.array(X_test)
                        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
                        x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
                        yhat = model.predict(x_input, verbose=0)
                        yhat = yhat + data[-1]
                        lst_output.append(yhat[0][0])
                        data.append(yhat[0][0])
                        i = i + 1
                    else:
                        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
                        x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
                        yhat = model.predict(x_input, verbose=0)
                        yhat = yhat + data[-1]
                        lst_output.append(yhat[0][0])
                        data.append(yhat[0][0])
                        i = i + 1
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                from io import StringIO
                import base64

                img = io.BytesIO()
                sns.set_style("dark")  # E.G.

                y = lst_output
                x = [x for x in range(len(lst_output))]

                plt.plot(x, y)
                plt.xlabel('Day count after 14/11/2020')
                plt.ylabel('MegaUnits')
                plt.savefig(img, format='png')
                plt.close()
                img.seek(0)

                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                print(plot_url)
                # gr = sns.lineplot(x=[x for x in range(len(lst_output))], y=lst_output)
                # import os
                # strFile = "static/image/pred.jpg"
                # if os.path.exists(strFile):
                #     os.remove(strFile)  # Opt.: os.system("rm "+strFile)
                # gr.figure.savefig(strFile)

            prediction1 = lst_output[-1]

            return render_template('results.html',prediction=prediction1,State_name=col_name,Date=d0.strftime("%m/%d/%Y"),plot_url=plot_url)



        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	main.run(debug=True) # running the app
