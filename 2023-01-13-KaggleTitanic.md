{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNlMv4Dzm0Ou7zZ9OiF9CIL",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rhkrtpgus/TIL/blob/main/2023_01_13_KaggleTitanic.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#**활동내용**"
      ],
      "metadata": {
        "id": "6dN8__USTDlD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###**케글 대회 소개**"
      ],
      "metadata": {
        "id": "hXhtM_8vUBQt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "타이타닉 생존자 분석\n",
        "[주소] - https://www.kaggle.com/competitions/titanic\n",
        "\n",
        "타이타닉 생존자 예측\n",
        "\n",
        "데이터: 타이타닉 탑증자의 여러 정보(예:나이, 성별, 탑승장소 등등)\n",
        "\n",
        "목적: 미리 제공된 약 800명의 데이터를 가지고 모델을 학습시켜 테스트 인원들의 생존여부 분석. 생존여부를 정확하게 예측한 정도 즉 정확도를 점차 높이는것이 궁극적인 목적이다."
      ],
      "metadata": {
        "id": "byKM8HXSv7x4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###**진행평가요소**\n",
        "\n",
        ">1. 데이터 전처리 방법\n",
        ">2. 모델 선정방법 및 이유\n",
        "\n",
        "###**피드백**\n",
        "개인피드백\n",
        "> 결측치와 각 열중에서 필요한것과 필요하지 않은 것들을 구분하여 상관관계 분석하고 어떤 것을 쓰고 쓰지 않을지 결정하여야한다.\n",
        "\n",
        "###**깃허브**\n",
        "> 대다수의 스터디 인원들이 깃허브 사용경험이 적어 이번 모각소 스터디를 계기로 깃허브 레포짓 및 블로그 작성하는 연습을 시행하였습니다. 이후 사용에 익숙해지면 깃허브 팀 레포짓을 만들어 팀단위로 대회에 나가는 증 해볼 계획입니다.\n",
        "\n",
        "#**활동사진**\n",
        "2시 활동사진\n",
        "\n",
        "![2시 활동사진](/images/2023-01-13-MGS1/2시 활동사진-1673591971027-1.png)\n",
        "\n",
        "3시 활동사진\n",
        "\n",
        "![3시 활동사진](/images/2023-01-13-MGS1/3시 활동사진.png)\n",
        "\n",
        "4시 활동사진\n",
        "\n",
        "![4시 활동사진](/images/2023-01-13-MGS1/4시 활동사진.png)\n",
        "\n",
        "5시 종료사진\n",
        "\n",
        "#**개인활동내역**\n",
        "\n",
        "###**개인 활동 블로그 url**\n",
        "*전장훈: https://jhwannabe.tistory.com/25\n",
        "\n",
        "*신재현: https://hectorsin.github.io/categories/#mgs\n",
        "\n",
        "*곽세현: https://rhkrtpgus.github.io\n",
        "\n",
        "*강성현: https://seong-hyeon-2.github.io\n",
        "\n",
        "*김수진: https://sujin7822.github.io/\n",
        "\n",
        "###**개인 깃 TIL**\n",
        "*전장훈: https://github.com/JHWannabe/TIL\n",
        "\n",
        "*신재현: https://github.com/HectorSin/TIL\n",
        "\n",
        "*곽세현: https://github.com/rhkrtpgus/TIL\n",
        "\n",
        "*강성현: https://github.com/seong-hyeon-2/TIL\n",
        "\n",
        "*김수진: https://github.com/sujin7822/TIL"
      ],
      "metadata": {
        "id": "9X-HaGLAUpqW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from sklearn.metrics import roc_auc_score\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from tensorflow.keras.wrappers.scikit_learn import KerasClassifier\n",
        "from keras.callbacks import EarlyStopping\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")"
      ],
      "metadata": {
        "id": "j8aZDYWX2-Lb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###* pandas 사용 이유 \n",
        ">##### - 데이터 분석에 특화된 자료구조를 제공하는 라이브러리 인데 스트레드시트 구조로 데이터를 다룰 수 있어서 가장 직관적이다.\n",
        "\n",
        "###* numpy 사용 이유\n",
        ">##### - 데이터의 대부분은 숫자 배열로 볼 수 있다. numpy는 비교적 빠른 연산을 지원하고 메모리를 효율적으로 사용한다\n",
        "\n",
        "###* matplotlib 사용 이유 (vs. seaborn)\n",
        ">##### - 막대그래프\n",
        ">##### - 선 그래프\n",
        ">##### - 산점도\n",
        ">##### - ETC 등의 시각화를 진행하기에 용이하다\n"
      ],
      "metadata": {
        "id": "nKLEQzdCxk3Y"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **1. 데이터 불러오기**"
      ],
      "metadata": {
        "id": "fXszSHkS0V1N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WrSOMM3M1mDX",
        "outputId": "11603717-c345-4e0e-9679-ca2ca8669d64"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df = pd.read_csv('/content/drive/MyDrive/All-in/titanic/traintitanic.csv')\n",
        "test_df = pd.read_csv('/content/drive/MyDrive/All-in/titanic/testtitanic.csv') \n",
        "\n",
        "# 데이터 요약\n",
        "train_df.head()"
      ],
      "metadata": {
        "id": "xU-zD1Wq0anO",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 285
        },
        "outputId": "d2cf64c4-07f2-483b-a53f-92d55b5f14bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d65184ba-bd7b-4faa-bb8f-aef44e072725\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d65184ba-bd7b-4faa-bb8f-aef44e072725')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d65184ba-bd7b-4faa-bb8f-aef44e072725 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d65184ba-bd7b-4faa-bb8f-aef44e072725');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 414
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 마지막 데이터\n",
        "train_df.tail()"
      ],
      "metadata": {
        "id": "-b5nMxcF9ew0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "08701371-a748-4e5f-baf3-66d589cca6a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     PassengerId  Survived  Pclass                                      Name  \\\n",
              "886          887         0       2                     Montvila, Rev. Juozas   \n",
              "887          888         1       1              Graham, Miss. Margaret Edith   \n",
              "888          889         0       3  Johnston, Miss. Catherine Helen \"Carrie\"   \n",
              "889          890         1       1                     Behr, Mr. Karl Howell   \n",
              "890          891         0       3                       Dooley, Mr. Patrick   \n",
              "\n",
              "        Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  \n",
              "886    male  27.0      0      0      211536  13.00   NaN        S  \n",
              "887  female  19.0      0      0      112053  30.00   B42        S  \n",
              "888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  \n",
              "889    male  26.0      0      0      111369  30.00  C148        C  \n",
              "890    male  32.0      0      0      370376   7.75   NaN        Q  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-80d782db-5ece-436f-9ab8-079897961f7b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>886</th>\n",
              "      <td>887</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>Montvila, Rev. Juozas</td>\n",
              "      <td>male</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>211536</td>\n",
              "      <td>13.00</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>887</th>\n",
              "      <td>888</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Graham, Miss. Margaret Edith</td>\n",
              "      <td>female</td>\n",
              "      <td>19.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>112053</td>\n",
              "      <td>30.00</td>\n",
              "      <td>B42</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>888</th>\n",
              "      <td>889</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
              "      <td>female</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>W./C. 6607</td>\n",
              "      <td>23.45</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>889</th>\n",
              "      <td>890</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Behr, Mr. Karl Howell</td>\n",
              "      <td>male</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>111369</td>\n",
              "      <td>30.00</td>\n",
              "      <td>C148</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>890</th>\n",
              "      <td>891</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Dooley, Mr. Patrick</td>\n",
              "      <td>male</td>\n",
              "      <td>32.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>370376</td>\n",
              "      <td>7.75</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-80d782db-5ece-436f-9ab8-079897961f7b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-80d782db-5ece-436f-9ab8-079897961f7b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-80d782db-5ece-436f-9ab8-079897961f7b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 415
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **2. 데이터 살펴보기**\n",
        "##### * 데이터의 형태와 크기를 알아보고 결측치를 파악하여 어떠한 형태로 가공할 것인지 방향을 정한다."
      ],
      "metadata": {
        "id": "15Xr2Dfo3YV9"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        ">### **2.1 데이터 딕셔너리**\n",
        ">> ##### survival = 생존여부 (0=no, 1=yes)\n",
        ">> ##### pclass = 사회, 경제적 지위 (1=1st, 2=2st, 3=3st)\n",
        ">> ##### sex = 성별\n",
        ">> ##### age = 나이\n",
        ">> ##### sibsp = 타이타닉호에 탑승한 형제, 자매 수\n",
        ">> ##### parch = 타이타닉호에 탑승한 부모, 자녀 수\n",
        ">> ##### ticket =  티켓 번호\n",
        ">> ##### fare = 탑승 요금\n",
        ">> ##### cabin = 탑승 요금\n",
        ">> ##### embarked = 탑승 지역(항구 위치)\n",
        "\n",
        ">### **2.2 결측치 파악**\n",
        ">>##### 데이터의 형태 알아보기기"
      ],
      "metadata": {
        "id": "n3xsDzrj6Rv1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.shape"
      ],
      "metadata": {
        "id": "S8B_mLQO7iRh",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a08b4c09-4690-480b-aee1-7e57f0c28c8a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 416
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_df.shape \n",
        "# 테스트 데이터는 훈련 데이터로 학습 시킨 모델을 통해 라벨링 해야하므로 survived 열이 빠져서 11개 인것"
      ],
      "metadata": {
        "id": "BlD7T8Th48ul",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b41ea670-8bec-467d-8b67-1b3e89b45e17"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(418, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 417
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# train 데이터의 정보 \n",
        "train_df.info()"
      ],
      "metadata": {
        "id": "s32Kv9398ezZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ed83d03c-f58f-4c50-bd56-1930b25c2ec8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# test 데이터의 정보\n",
        "test_df.info()"
      ],
      "metadata": {
        "id": "qFtzgq-n8Hxe",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cfb5ec79-ea64-44c0-ddd6-64ad8e6b1048"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 418 entries, 0 to 417\n",
            "Data columns (total 11 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  418 non-null    int64  \n",
            " 1   Pclass       418 non-null    int64  \n",
            " 2   Name         418 non-null    object \n",
            " 3   Sex          418 non-null    object \n",
            " 4   Age          332 non-null    float64\n",
            " 5   SibSp        418 non-null    int64  \n",
            " 6   Parch        418 non-null    int64  \n",
            " 7   Ticket       418 non-null    object \n",
            " 8   Fare         417 non-null    float64\n",
            " 9   Cabin        91 non-null     object \n",
            " 10  Embarked     418 non-null    object \n",
            "dtypes: float64(2), int64(4), object(5)\n",
            "memory usage: 36.0+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# train 데이터셋 설명\n",
        "train_df.describe()"
      ],
      "metadata": {
        "id": "RyS0H0vL980r",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "a44d027e-2686-4eb4-ce2c-9a7b700c4a14"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
              "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
              "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
              "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
              "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
              "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
              "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
              "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
              "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
              "\n",
              "            Parch        Fare  \n",
              "count  891.000000  891.000000  \n",
              "mean     0.381594   32.204208  \n",
              "std      0.806057   49.693429  \n",
              "min      0.000000    0.000000  \n",
              "25%      0.000000    7.910400  \n",
              "50%      0.000000   14.454200  \n",
              "75%      0.000000   31.000000  \n",
              "max      6.000000  512.329200  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-406c5504-fe16-4536-b5ea-3f802bef36a1\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>29.699118</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>32.204208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>257.353842</td>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>14.526497</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>49.693429</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.420000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>223.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.910400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.454200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>668.500000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>31.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-406c5504-fe16-4536-b5ea-3f802bef36a1')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-406c5504-fe16-4536-b5ea-3f802bef36a1 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-406c5504-fe16-4536-b5ea-3f802bef36a1');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 420
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">##### * 훈련 자료의 샘플 수: 891\n",
        ">##### * 훈련 자료 생존 내 생존율: 38.4%"
      ],
      "metadata": {
        "id": "CQEiTKia143o"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# null의 갯수 세어보기\n",
        "train_df.isnull().sum()"
      ],
      "metadata": {
        "id": "sGfbOfz-8cno",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a0a45773-b98c-4a16-a9d2-35b424a1af4d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Survived         0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age            177\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             0\n",
              "Cabin          687\n",
              "Embarked         2\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 421
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_df.isnull().sum()"
      ],
      "metadata": {
        "id": "xLyfuUOm9GGS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dc1b3175-eb31-4eda-b7e3-310fb15d2bf2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age             86\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             1\n",
              "Cabin          327\n",
              "Embarked         0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 422
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">>##### null 값이 있으면 모델이 안돌아간다. 정확한 갯수를 세는 이유는 결측치를 없애기 위함이고, 결측치가 너무 많으면 어떤것으로 대체해야하는지 알기 위해서\n",
        "\n",
        ">>##### 반 이상이 결측치다 -> 평균값으로 대체하는거보다 결측치로 대체\n",
        ">>##### 소규모 -> 평균값으로 대체하거나 데이터에 특성에 따라서 처리한다"
      ],
      "metadata": {
        "id": "q195xgMfGGse"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Sex에 따른 생존률\n",
        "train_df.groupby(['Sex'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "dyAWSD6k04Fk",
        "outputId": "f26d9728-3a55-4f8b-95ef-f8d3f8bbe91c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc10ebcef10>"
            ]
          },
          "metadata": {},
          "execution_count": 423
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEGCAYAAAB1iW6ZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dd3QVdf7G8fcnlY6UgEjvCEgNHRLdpaug2EAEK4iAQOLadlfXtruu7oaiIGLBLiAqgoIBLAkdEuk9IEgTgiC9yvf3B3F/kQ0S4Ia55Xmdk3OYmW/ufO7xnOeMczPPNeccIiIS+MK8HkBERHxDgS4iEiQU6CIiQUKBLiISJBToIiJBIsKrE5csWdJVqlTJq9OLiASk9PT03c65mJyOeRbolSpVIi0tzavTi4gEJDPbfLZjuuUiIhIkFOgiIkFCgS4iEiQU6CIiQUKBLiISJBToIiJBQoEuIhIkAi7QN2Qe5D/T13L0xC9ejyIi4lcCLtBnrNrJS19ncO2IWaRv3uP1OCIifiPgAr1ffFXevqcpR0+c4ubR83hq8koOHTvp9VgiIp4LuEAHiK8RQ3JCHL2bV+TteZtoPzSV1HWZXo8lIuKpgAx0gELRETzdtS4T7m9BdGQYvd9cyJ8+WsrPh497PZqIiCcCNtB/1aRScaYOakP/q6vy6eJttE1KZdryHV6PJSJyyQV8oAPkiwznkY61+GxAK0oVjuaB97/jgffS2XXgqNejiYhcMkER6L+qW7Yonw1sxSMda/LVml20S0rlo7QtOOe8Hk1EJM8FVaADRIaH0f/qakwb3IYapQvx8MRl9H5zIVv2HPZ6NBGRPBV0gf6rqjGFGN+3Bc90rcN3m/fSYVgqb835nlOndLUuIsEpaAMdICzM6N2iEskJccRWKs5TU1Zxy6vzyNh1wOvRRER8LqgD/VflihXg7bub8J9b6pOx6yCdh89m5DcZnPjllNejiYj4TEgEOoCZcVPjcsxMjKdt7VK8mLyWri/PYcW2fV6PJiLiE7kKdDPraGZrzSzDzB7L4fhQM1uS9bPOzH72/ai+EVM4mlE9GzP6jsZkHjxG15Fz+NeXa1T2JSIBL+JcC8wsHBgJtAO2AovMbLJzbtWva5xzCdnWPwg0zINZfapj3ctpUaUEf5+6ile+3UDyih95/qZ6NK1c3OvRREQuSG6u0JsCGc65jc6548A4oOvvrO8BfOiL4fJa0QKRvHBzfd67txnHfznFra/O44lJKziosi8RCUC5CfSywJZs21uz9v0PM6sIVAa+PsvxvmaWZmZpmZn+U6bVunpJkofEcXerSry3YDPtk1L4Zu0ur8cSETkvvv5QtDsw0TmX4w1p59wY51yscy42JibGx6e+OAWjI/jb9XWY2K8lBaIjuHvsIhLHL2HvIZV9iUhgyE2gbwPKZ9sul7UvJ90JkNstZ9O4YjG+GNSaB/9QjclLt9NuaApfLNuh+gAR8Xu5CfRFQHUzq2xmUZwO7clnLjKzWkAxYJ5vR7z0oiPCeah9TSYPbE2ZovkZ8MF33P9uOrv2q+xLRPzXOQPdOXcSGAgkA6uBCc65lWb2jJl1yba0OzDOBdGlbO0rivBp/5Y83qkWKesy+WNSChMWqexLRPyTeRVOsbGxLi0tzZNzX4iNmQd57JPlLPx+D62qleCfN9ajQokCXo8lIiHGzNKdc7E5HQuZJ0UvVpWYQozr05znbqjL0i376DAslTdmf88vKvsSET+hQD8PYWHGHc0rMj0hjmZVivPs56u4efRc1u9U2ZeIeE+BfgGuuCw/Y+9qwrDbGrBp9yGuHTGbEV+t5/hJlX2JiHcU6BfIzLihYVlmJMbToe7lJM1YR5eXZ7Nsq9/W2IhIkFOgX6SShaJ5qUdDXusdy97Dx7lh5Bz+OXU1R46r7EtELi0Fuo+0q12a6Qnx3NakPK+mbqTT8FTmb/zJ67FEJIQo0H2oaP5I/tmtHh/c14xTDrqPmc9fPl3OgaMnvB5NREKAAj0PtKxWki+HtOG+1pX5cOEPtB+aytdrdno9logEOQV6HikQFcFfr6vNxw+0pHC+CO55K40h4xazR2VfIpJHFOh5rGGFYnz+YBsG/7E6XyzfQdukFCYv3a76ABHxOQX6JRAVEUZCuxpMebA15YvlZ9CHi+nzTjo/7lPZl4j4jgL9Eqp1eRE+6d+Kv3S+ktkZmbRLSuHDhT/oal1EfEKBfomFhxl94qrw5eA46pQtwuOfLOf21xaw+adDXo8mIgFOge6RSiUL8sF9zflnt6tYse102dfrszaq7EtELpgC3UNhYUaPphWYkRhP62olee6L1XR7ZS5rf1TZl4icPwW6H7i8aD5e6x3LiB4N2bLnMNe9NIuhM9ap7EtEzosC3U+YGV3qX8HMxHg6X1WG4V+t57qXZrFki8q+RCR3FOh+pnjBKIZ3b8gbd8ay/8hJuo2aw3Ofr1LZl4ickwLdT/3xytJMT4yje9MKvD77ezoMS2Xuht1ejyUifkyB7seK5IvkHzdexYd9mhNmcPtrC3j8k2XsV9mXiORAgR4AWlQtwbTBcdwfV4Xxi7bQLimFmatU9iUiv6VADxD5o8J5vPOVTBrQimIForjvnTQe/HAxuw8e83o0EfETCvQAU6/cZUwe2JrEdjX4csUO2iWlMGnxNtUHiIgCPRBFRYQx6I/V+WJQGyqWKMiQ8Uu49+00tv98xOvRRMRDCvQAVqN0YT5+oCVPXFebeRt+ov3QVN6bv5lTqg8QCUkK9AAXHmbc27oyyUPiqF++KH+dtIIer83n+90q+xIJNQr0IFGhRAHeu7cZL9xUj1U79tNxWCqvpmzg5C+qDxAJFbkKdDPraGZrzSzDzB47y5pbzWyVma00sw98O6bkhplxa5PyzEyMJ65GDP+ctoYbR81l1fb9Xo8mIpfAOQPdzMKBkUAnoDbQw8xqn7GmOvA40Mo5VwcYkgezSi6VLpKPMb0aM/L2RuzYd4QuL8/mP9PXcuyk6gNEgllurtCbAhnOuY3OuePAOKDrGWv6ACOdc3sBnHO7fDumnC8z49p6ZZiREE+X+lfw0tcZXDtiNumb93o9mojkkdwEellgS7btrVn7sqsB1DCzOWY238w65vRCZtbXzNLMLC0zM/PCJpbzUqxgFEm3NWDs3U04fOwkN4+ey9NTVnL4+EmvRxMRH/PVh6IRQHXgaqAH8JqZXXbmIufcGOdcrHMuNiYmxkenlty4pmYppifG06t5RcbO2UT7oanMXq+yL5FgkptA3waUz7ZdLmtfdluByc65E86574F1nA548SOFoiN4pmtdJtzfgsjwMO54YwGPTFzKviMq+xIJBrkJ9EVAdTOrbGZRQHdg8hlrJnH66hwzK8npWzAbfTin+FDTysWZNrgND1xdlY+/20a7pBSSV/7o9VgicpHOGejOuZPAQCAZWA1McM6tNLNnzKxL1rJk4CczWwV8AzzsnPspr4aWi5cvMpxHO9ZiUv9WlCgUzf3vpjPg/e/IPKCyL5FAZV6VOsXGxrq0tDRPzi2/deKXU4xJ3cjwmevJHxXOk9fVplujspiZ16OJyBnMLN05F5vTMT0pKkSGhzHgmmpMHdyaaqUK8dBHS7lr7CK2qexLJKAo0OW/qpUqzEf3t+Cp62uzaNMe2iel8M68TSr7EgkQCnT5jbAw465Wp8u+GlUsxpOfreS2MfPYkHnQ69FE5BwU6JKj8sUL8M49TXnx5nqs/fEAnYbPYtS3GZxQ2ZeI31Kgy1mZGbfElmfmQ/H8oWYpXvhyLTeMnMOKbfu8Hk1EcqBAl3MqVTgfo3s15pWejdi5/xhdR87hxeQ1HD2hsi8Rf6JAl1zrdFUZZibGcWPDsoz8ZgOdR8wibdMer8cSkSwKdDkvlxWI4t+31Oede5py7MQpbnl1Hk9NXsmhYyr7EvGaAl0uSFyNGKYnxHFni0q8Pe902VfKOjVoinhJgS4XrGB0BE91qcNH97cgOjKMO99cyEMTlvLz4eNejyYSkhToctFiKxVn6qA2DLimKpOWbKNtUirTlu/weiyRkKNAF5/IFxnOwx1qMXlgK0oXieaB97+j37vp7Np/1OvRREKGAl18qs4VRflsQCse7ViLr9fuom1SCh+lbcGrEjiRUKJAF5+LCA/jgaurMm1wG2peXpiHJy6j95sL2bLnsNejiQQ1BbrkmaoxhRjftwXPdq3Dd5v30mFYKm/N+Z5fVPYlkicU6JKnwsKMXi0qkZwQR5NKxXlqyipufXUeGbsOeD2aSNBRoMslUa5YAd66uwlJt9ZnQ+ZBOg+fzctfr1fZl4gPKdDlkjEzujUqx4yEeNrVKc2/p6+jy8sq+xLxFQW6XHIxhaMZeXsjXu3VmN0HT5d9PT9NZV8iF0uBLp7pUOdyZibEc3OjcoxO2UDn4bNY+L3KvkQulAJdPFW0QCT/urke793bjOO/nOLWV+fxxKQVHDh6wuvRRAKOAl38QuvqJZmeEMc9rSrz3oLNdBiayjdrd3k9lkhAUaCL3ygQFcGT19dmYr+WFIyO4O6xi0gcv4S9h1T2JZIbCnTxO40rFuPzQa0Z9IdqTF66nbZJKXy+bLvqA0TOQYEufik6IpzE9jWZ8mBrrrgsPwM/WMz976azU2VfImelQBe/dmWZInzavyWPd6pFyrpM2ialMH7RD7paF8mBAl38XkR4GPfHV+XLIXFcWaYIj368nJ6vL+CHn1T2JZJdrgLdzDqa2VozyzCzx3I4fpeZZZrZkqyf+3w/qoS6yiULMq5Pc/5+Y12Wbd1Hh2GpvDFbZV8ivzpnoJtZODAS6ATUBnqYWe0clo53zjXI+nndx3OKAKfLvno2q8iMxDhaVC3Bs5+v4qZX5rJup8q+RHJzhd4UyHDObXTOHQfGAV3zdiyR31emaH7euDOW4d0bsPmnQ1w7YhYjvlrP8ZMq+5LQlZtALwtsyba9NWvfmW4ys2VmNtHMyuf0QmbW18zSzCwtM1PfEC8Xx8zo2qAsMxPj6Vi3DEkz1tHl5dks3fKz16OJeMJXH4pOASo55+oBM4C3c1rknBvjnIt1zsXGxMT46NQS6koUiualHg15rXcsew8f58ZRc/jH1NUcOa6yLwktuQn0bUD2K+5yWfv+yzn3k3PuWNbm60Bj34wnknvtapdmRmI8tzUpz5jUjXQansq8DT95PZbIJZObQF8EVDezymYWBXQHJmdfYGZlsm12AVb7bkSR3CuSL5J/dqvHB/c145SDHq/N58+fLme/yr4kBJwz0J1zJ4GBQDKng3qCc26lmT1jZl2ylg0ys5VmthQYBNyVVwOL5EbLaiVJHhJHnzaVGbfwB9onpfL1mp1ejyWSp8yrJ+5iY2NdWlqaJ+eW0LJky888OnEZa3ceoGuDK3jyutqUKBTt9VgiF8TM0p1zsTkd05OiEvQalL+MKQ+2Zkjb6kxdvoN2Q1OZvFRlXxJ8FOgSEqIiwhjStgafP9iG8sULMOjDxfR5J40d+454PZqIzyjQJaTUvLwwnzzQkr9eeyWzM3bTPimVDxb8wCnVB0gQUKBLyAkPM+5rU4XkIXHULVuUP3+6nNtfn8+m3Ye8Hk3koijQJWRVLFGQD/o04/luV7Fy2346Dk/ltdSNKvuSgKVAl5BmZnRvWoEZifG0rlaSv09dTbdRc1j7o8q+JPAo0EWAy4vm47XesbzUoyFb9x7hupdmMXTGOo6dVH2ABA4FukgWM+P6+lcwIzGea68qw/Cv1nP9S7NZ/MNer0cTyRUFusgZiheMYlj3hrx5VywHjp6k2ytzefbzVRw+ftLr0UR+lwJd5Cz+UKs00xPi6NmsAm/M/p6Ow2YxN2O312OJnJUCXeR3FM4XyXM3XMW4vs0JM7j99QU89vEy9h1R2Zf4HwW6SC40r1KCL4fEcX98FSakbaH90BRmrFLZl/gXBbpILuWLDOfxTlcyaUArihWIos87aQz84Dt2Hzx27l8WuQQU6CLnqV65y5g8sDUPtavB9JU7aZuUwqeLt6rsSzynQBe5AFERYTz4x+p8Mag1lUsWJGH8Uu55axHbf1bZl3hHgS5yEaqXLszEfi158rrazN+4h/ZDU3l3/maVfYknFOgiFyk8zLindWWmJ8TRoPxlPDFpBd1fm8/3KvuSS0yBLuIj5YsX4N17m/LCTfVYvWM/HYelMjplAyd/OeX1aBIiFOgiPmRm3NqkPDMT44mvEcPz09Zw46i5rNq+3+vRJAQo0EXyQOki+Xi1V2NG9WzEjn1H6PLybP4zfa3KviRPKdBF8oiZ0fmqMsxIiKdLgyt46esMrh0xm/TNKvuSvKFAF8ljxQpGkXRrA966uwlHjv/CzaPn8vSUlRw6prIv8S0FusglcnXNUiQnxNGreUXGztlEh2GpzFqf6fVYEkQU6CKXUKHoCJ7pWpcJ97cgKjyMXm8s5JGJS9l3WGVfcvEU6CIeaFq5OFMHt+GBq6vy8XfbaDs0hS9X/Oj1WBLgFOgiHskXGc6jHWvx2YBWxBSKpt976fR/P51dB456PZoEqFwFupl1NLO1ZpZhZo/9zrqbzMyZWazvRhQJbnXLFuWzga14uENNZq7eRbukVD5OV9mXnL9zBrqZhQMjgU5AbaCHmdXOYV1hYDCwwNdDigS7yPAwBlxTjamD2lCtVCEe+mgpd45dxNa9h70eTQJIbq7QmwIZzrmNzrnjwDigaw7rngX+Bej/F0UuULVShfjo/hY83aUOaZv20GFoKu/M26SyL8mV3AR6WWBLtu2tWfv+y8waAeWdc1/83guZWV8zSzOztMxM/bmWSE7Cwow7W1YieUgcjSoW48nPVnLbmHlsyDzo9Wji5y76Q1EzCwOSgIfOtdY5N8Y5F+uci42JibnYU4sEtfLFC/DOPU359y31WbfzIJ2Gz2LkNxmcUNmXnEVuAn0bUD7bdrmsfb8qDNQFvjWzTUBzYLI+GBW5eGbGzY3LMSMxjrZXluLF5LXcMHIOK7bt83o08UO5CfRFQHUzq2xmUUB3YPKvB51z+5xzJZ1zlZxzlYD5QBfnXFqeTCwSgkoVzseono0ZfUcjdu4/RteRc3jhyzUcPaGyL/l/5wx059xJYCCQDKwGJjjnVprZM2bWJa8HFJH/17FuGb5KjKdbw7KM+nYDnUfMIm3THq/HEj9hXv2ta2xsrEtL00W8yIVKXZfJ458sZ/u+I/RuXpGHO9aiUHSE12NJHjOzdOdcjre09aSoSICKqxHD9IQ47mxRiXfmb6bD0FRS1umvx0KZAl0kgBWMjuCpLnWY2K8F+SLDuPPNhSROWMLPh497PZp4QIEuEgQaVyzOF4PaMPCaakxesp22SSlMXb7D67HkElOgiwSJfJHh/KlDTT4b2IrLi+aj//vf0e/ddHbt18PboUKBLhJk6lxRlEn9W/Fox1p8vXYXbZNSmJC2RWVfIUCBLhKEIsLDeODqqnw5uA21Li/CIxOX0fvNhWzZo7KvYKZAFwliVWIKMa5vc57tWofvNu+lw7BUxs75nl9U9hWUFOgiQS4szOjVohLTE+NpWrk4T09ZxS2j55Kx64DXo4mPKdBFQkTZy/Iz9q4mDL2tPht3H6Lz8Nm8/PV6lX0FEQW6SAgxM25sWI6ZifG0q1Oaf09fx/UvzWb5VpV9BQMFukgIKlkompG3N+LVXo3Zc+g4N4yaw/PTVPYV6BToIiGsQ53LmZEYz82NyjE6ZQOdhs9iwcafvB5LLpACXSTEFc0fyb9ursf79zXj5KlT3DZmPn+dtJwDR094PZqcJwW6iADQqlpJkofEcW/ryry/4Ac6DE3lmzW7vB5LzoMCXUT+q0BUBE9cV5uPH2hJwegI7n5rEQnjl7DnkMq+AoECXUT+R6MKxfh8UGsG/bE6U5Zup11SCp8v2676AD+nQBeRHEVHhJPYrgZTHmxN2WL5GfjBYvq+m85OlX35LQW6iPyuK8sU4ZMHWvLnzrVIXZdJ26QUxi38QVfrfkiBLiLnFBEeRt+4qiQPiaN2mSI89slyer6+gB9+UtmXP1Ggi0iuVSpZkA/7NOcfN17Fsq37aD8shddnbVTZl59QoIvIeQkLM25vVoEZiXG0rFqS575YzU2vzGXdTpV9eU2BLiIXpEzR/LxxZyzDuzfghz2HuXbELIbPXM/xkyr78ooCXUQumJnRtUFZZiTE0aluGYbOXEeXl2ezdMvPXo8WkhToInLRShSKZkSPhrzeO5afD5/gxlFz+MfU1Rw5rrKvS0mBLiI+07Z2aaYnxtG9aQXGpG6k4/BU5m1Q2delokAXEZ8qki+Sf9x4FR/0aQZAj9fm8/gny9mvsq88p0AXkTzRsmpJvhwcR9+4Koxf9APtk1L5avVOr8cKarkKdDPraGZrzSzDzB7L4Xg/M1tuZkvMbLaZ1fb9qCISaPJHhfPnzlfySf9WFM0fyb1vpzHow8X8dPCY16MFpXMGupmFAyOBTkBtoEcOgf2Bc+4q51wD4AUgyeeTikjAalD+MqY82JqEtjWYtmIH7Yam8tmSbaoP8LHcXKE3BTKccxudc8eBcUDX7Aucc/uzbRYE9F9JRH4jKiKMwW2r88WgNlQoXoDB45Zw39tp7Nh3xOvRgkZuAr0ssCXb9tasfb9hZgPMbAOnr9AH5fRCZtbXzNLMLC0zM/NC5hWRAFejdGE+fqAlf732SuZs2E27pFTeX7CZU6oPuGg++1DUOTfSOVcVeBT461nWjHHOxTrnYmNiYnx1ahEJMOFhxn1tqjB9SDz1yhXlL5+u4PbX57Np9yGvRwtouQn0bUD5bNvlsvadzTjghosZSkRCQ4USBXj/vmY83+0qVm7bT4dhqYxJ3cDJX1QfcCFyE+iLgOpmVtnMooDuwOTsC8yserbNa4H1vhtRRIKZmdG9aQVmJMbTpnoM/5i6hptemcuaH/ef+5flN84Z6M65k8BAIBlYDUxwzq00s2fMrEvWsoFmttLMlgCJwJ15NrGIBKXLi+bjtd6Nefn2hmzde4TrRswmacY6jp1UfUBumVd/NhQbG+vS0tI8ObeI+Le9h47zzOer+HTxNmqULsS/bqpHwwrFvB7LL5hZunMuNqdjelJURPxOsYJRDL2tAWPvasKBoyfp9spcnv18FYePn/R6NL+mQBcRv3VNrVJMT4ijZ7MKvDH7ezoMS2VOxm6vx/JbCnQR8WuF80Xy3A1XMb5vcyLCwuj5+gIe+3gZ+46o7OtMCnQRCQjNqpRg2uA23B9fhQlpW2iXlML0lT96PZZfUaCLSMDIFxnO452uZNKAVhQvGEXfd9MZ+MF37FbZF6BAF5EAVK/c6bKvP7WvwfSVO2mblMKni7eGfNmXAl1EAlJkeBgD/1CdqYNbU6VkQRLGL+Xutxax7efQLftSoItIQKtWqjAf9WvJ366vzYKNe2iflMK780Oz7EuBLiIBLzzMuLtVZaYnxNGwQjGemLSC7mPmszHzoNejXVIKdBEJGuWLF+Dde5vyws31WPPjfjoNn8XolNAp+1Kgi0hQMTNujS3PzMR4rq4Zw/PT1nDDqDms2h78ZV8KdBEJSqWK5OPVXrG80rMRP+47RpeXZ/Pv5LUcPRG8ZV8KdBEJap2uKsPMxDi6NijLy99kcO2IWaRv3uP1WHlCgS4iQe+yAlH859b6vH1PU46eOMXNo+fx1OSVHDoWXGVfCnQRCRnxNWJIToijd/OKvDV3Ex2GpTJrffB8v7ECXURCSqHoCJ7uWpeP+rUgKiKMXm8s5OGPlrLvcOCXfSnQRSQkNalUnKmD2tD/6qp8sngbbYem8OWKHV6PdVEU6CISsvJFhvNIx1p8NqAVMYWi6ffedzzwXjq7Dhz1erQLokAXkZBXt2xRPhvYioc71OSrNbtol5TKxPTAK/tSoIuIcLrsa8A11Zg6qA3VSxXiTx8t5c6xi9i697DXo+WaAl1EJJtqpQox4f4WPN2lDmmb9tB+aCpvz90UEGVfCnQRkTOEhRl3tqzE9IQ4YisV52+TV3Lrq/PI2OXfZV8KdBGRsyhXrABv392E/9xSn/W7DtJ5+CxGfpPBCT8t+1Kgi4j8DjPjpsblmJkYT9vapXgxeS1dX57Dim37vB7tfyjQRURyIaZwNKN6Nmb0HY3IPHiMriPn8K8v1/hV2ZcCXUTkPHSsW4aZCfF0a1iWV77dQOfhs1i0yT/KvhToIiLnqWiBSF68pT7v3tuU47+c4pbR83jysxUc9LjsK1eBbmYdzWytmWWY2WM5HE80s1VmtszMvjKzir4fVUTEv7SpHkPykDjublWJd+dvpsPQVL5du8uzec4Z6GYWDowEOgG1gR5mVvuMZYuBWOdcPWAi8IKvBxUR8UcFoyP42/V1mNivJfmjwrlr7CISJyxh76Hjl3yW3FyhNwUynHMbnXPHgXFA1+wLnHPfOOd+fZxqPlDOt2OKiPi3xhWL8cWg1jz4h2pMXrKddkNTmLp8xyWtD8hNoJcFtmTb3pq172zuBabldMDM+ppZmpmlZWYGTwexiAhAdEQ4D7WvyeSBrSlTND/93/+Ofu+ls2v/pSn78umHomZ2BxALvJjTcefcGOdcrHMuNiYmxpenFhHxG7WvKMKn/VvyWKdafLs2k7ZJKUxI25LnV+u5CfRtQPls2+Wy9v2GmbUF/gJ0cc4d8814IiKBKSI8jH7xVZk2uA21yhThkYnL6PXGQrbsybuyr9wE+iKguplVNrMooDswOfsCM2sIvMrpMPfuI14RET9TJaYQ4/o057kb6rJky8+0H5rKlKXb8+Rc5wx059xJYCCQDKwGJjjnVprZM2bWJWvZi0Ah4CMzW2Jmk8/yciIiIScszLijeUWmJ8TRqlpJKpcsmCfnMa8K3GNjY11aWpon5xYRCVRmlu6ci83pmJ4UFREJEgp0EZEgoUAXEQkSCnQRkSChQBcRCRIKdBGRIKFAFxEJEgp0EZEg4dmDRWaWCWy+wF8vCez24TiBQO85NOg9h4aLec8VnXM5tht6FugXw8zSzvakVLDSew4Nes+hIa/es265iIgECQW6iEiQCNRAH+P1AB7Qew4Nes+hIU/ec0DeQxcRkf8VqFfoIiJyBgW6iEiQ8CTQzWyQma02s/fz6PWfMrM/5QJjDO8AAAL+SURBVMVri4j4qwiPztsfaOuc2+rR+UVEgs4lD3QzGw1UAaaZ2TigKlAXiASecs59ZmZ3ATcABYHqwL+BKKAXcAzo7JzbY2Z9gL5ZxzKAXs65w2ecryowEogBDgN9nHNr8vyNiohcYpf8lotzrh+wHbiG04H9tXOuadb2i2b267en1gW6AU2AvwOHnXMNgXlA76w1nzjnmjjn6nP6C6zvzeGUY4AHnXONgT8Bo/LmnYmIeMurWy6/ag90yXa/Ox9QIevf3zjnDgAHzGwfMCVr/3KgXta/65rZc8BlQCEgOfuLm1khoCXwkZn9ujs6L96IiIjXvA50A25yzq39zU6zZpy+tfKrU9m2T/H/c78F3OCcW5p1m+bqM14/DPjZOdfAt2OLiPgfr/9sMRl40LIun82s4Xn+fmFgh5lFAj3PPOic2w98b2a3ZL2+mVn9i5xZRMQveR3oz3L6w9BlZrYya/t8PAEsAOYAZ/ugsydwr5ktBVYCXS9wVhERv6ZH/0VEgoTXV+giIuIjCnQRkSChQBcRCRIKdBGRIKFAFxEJEgp0CUlm9hczW2lmy8xsSdbDbCIBzesnRUUuOTNrAVwHNHLOHTOzkpwueBMJaLpCl1BUBtjtnDsG4Jzb7ZzbbmaNzSzFzNLNLNnMyphZUTNba2Y1Aczsw6yWTxG/oweLJORklbbNBgoAM4HxwFwgBejqnMs0s9uADs65e8ysHfAMMBy4yznX0aPRRX6XbrlIyHHOHTSzxkAbTtc2jwee43Rl84ysaqFwYEfW+hlZfUAjAXUBid/SFbqEPDO7GRgA5HPOtcjheBinr94rcfrLVZZf2glFckf30CXkmFlNM6uebVcDTn9BSkzWB6aYWaSZ1ck6npB1/HZgbFa7p4jf0RW6hJys2y0vcfqLUU5y+usL+wLlgBFAUU7fjhwGpAKTgKbOuQNmlgQccM79zYvZRX6PAl1EJEjolouISJBQoIuIBAkFuohIkFCgi4gECQW6iEiQUKCLiAQJBbqISJD4P25gE+0m6i7JAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "값이 나오지 않아 female을 0, male을 1로 할당하여 다시 해보기로 했다"
      ],
      "metadata": {
        "id": "8lU2f2FL1Ltk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 285
        },
        "id": "ix3ioaN32060",
        "outputId": "3dde0d05-ddcc-486b-e304-c11282f93b30"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-24eb8bf2-fdc2-455a-af70-65c074932f08\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-24eb8bf2-fdc2-455a-af70-65c074932f08')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-24eb8bf2-fdc2-455a-af70-65c074932f08 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-24eb8bf2-fdc2-455a-af70-65c074932f08');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 424
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_and_test = [train_df, test_df]"
      ],
      "metadata": {
        "id": "q-nH9XhX3GkJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for dataset in train_and_test:\n",
        "    dataset['Sex'] = dataset['Sex'].replace('female', 0)\n",
        "    dataset['Sex'] = dataset['Sex'].replace(\"male\", 1)\n",
        "\n",
        "train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        },
        "id": "bb2qdzVN1vhY",
        "outputId": "1d7e7bf1-3aa5-43ae-df1f-04be8419a056"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Sex  Survived\n",
              "0    0  0.742038\n",
              "1    1  0.188908"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b4950148-8d6e-48af-b8af-430da93608a5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.742038</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.188908</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b4950148-8d6e-48af-b8af-430da93608a5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b4950148-8d6e-48af-b8af-430da93608a5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b4950148-8d6e-48af-b8af-430da93608a5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 426
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Embarked도 문자가 아닌 숫자로 바꿔주기\n",
        "for dataset in train_and_test:\n",
        "    dataset['Embarked'] = dataset['Embarked'].replace('S', 0)\n",
        "    dataset['Embarked'] = dataset['Embarked'].replace('C', 1)\n",
        "    dataset['Embarked'] = dataset['Embarked'].replace('Q', 2)\n",
        "\n",
        "# Embarked에 따른 생존률\n",
        "train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "id": "hrIsj3Xa6rbF",
        "outputId": "071a9b6a-9ea0-430b-cfe0-b156499856b2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Embarked  Survived\n",
              "0       0.0  0.336957\n",
              "1       1.0  0.553571\n",
              "2       2.0  0.389610"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-bc705749-a411-4848-a261-63abc850a3b3\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Embarked</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.336957</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1.0</td>\n",
              "      <td>0.553571</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2.0</td>\n",
              "      <td>0.389610</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-bc705749-a411-4848-a261-63abc850a3b3')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-bc705749-a411-4848-a261-63abc850a3b3 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-bc705749-a411-4848-a261-63abc850a3b3');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 427
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Age도 범주형 데이터로 바꿔주기\n",
        "\n",
        "# 영유아~아동(0~12): 0\n",
        "# 청소년~청년(13~18): 1\n",
        "# 청년~중년(19~49) : 2\n",
        "# 장년~노년(50~64) : 3\n",
        "# 그 이상 : 4\n",
        "\n",
        "for dataset in train_and_test:\n",
        "    dataset.loc[dataset['Age'] <= 12, 'Age'] = 0\n",
        "    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age'] = 1\n",
        "    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 49), 'Age'] = 2\n",
        "    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 64), 'Age'] = 3\n",
        "    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4"
      ],
      "metadata": {
        "id": "WxM4C3uSA4yD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head(10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "id": "FPbRE0A4CAsJ",
        "outputId": "aa0ca138-b1b8-45e2-be6b-870d3f06600f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex   Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    1   2.0      1      0   2.0000      0.0\n",
              "1         1       1    0   0.0      1      0  71.2833      1.0\n",
              "2         1       3    0   2.0      0      0   2.0000      0.0\n",
              "3         1       1    0   2.0      1      0   1.0000      0.0\n",
              "4         0       3    1   2.0      0      0   2.0000      0.0\n",
              "5         0       3    1  29.0      0      0   2.0000      2.0\n",
              "6         0       1    1   3.0      0      0   1.0000      0.0\n",
              "7         0       3    1   0.0      3      1   2.0000      0.0\n",
              "8         1       3    0   2.0      0      2   2.0000      0.0\n",
              "9         1       2    0   1.0      1      0   2.0000      1.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b18e0d80-5794-44f3-bc84-7accb400751e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>29.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>2.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>0.0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b18e0d80-5794-44f3-bc84-7accb400751e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b18e0d80-5794-44f3-bc84-7accb400751e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b18e0d80-5794-44f3-bc84-7accb400751e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 483
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.groupby(['Sex'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "uP2Ucix_3MMa",
        "outputId": "f6bb5cad-136e-4d05-9257-685a7110eda6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc10ebbbd00>"
            ]
          },
          "metadata": {},
          "execution_count": 429
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df[[\"Sex\", \"Survived\"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        },
        "id": "Pgjnsxz11U96",
        "outputId": "2c92a047-d068-4ef1-f735-869a12789e6f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Sex  Survived\n",
              "0    0  0.742038\n",
              "1    1  0.188908"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-433ed1e2-5b70-4160-8244-037eb3726aa1\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.742038</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.188908</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-433ed1e2-5b70-4160-8244-037eb3726aa1')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-433ed1e2-5b70-4160-8244-037eb3726aa1 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-433ed1e2-5b70-4160-8244-037eb3726aa1');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 430
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#PClass에 따른 생존률\n",
        "train_df.groupby(['Pclass'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "elIaKiapyIkt",
        "outputId": "168d970a-5b03-4a00-b4fd-ffdae59ed20b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc115c9a250>"
            ]
          },
          "metadata": {},
          "execution_count": 431
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "id": "zEVmIhuHzAY0",
        "outputId": "1ec03a2e-106a-4a2e-9168-20457762c0bc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Pclass  Survived\n",
              "0       1  0.629630\n",
              "1       2  0.472826\n",
              "2       3  0.242363"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fd6d28ef-968e-48ff-93ae-b3da1abfe051\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0.629630</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>0.472826</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>0.242363</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fd6d28ef-968e-48ff-93ae-b3da1abfe051')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-fd6d28ef-968e-48ff-93ae-b3da1abfe051 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-fd6d28ef-968e-48ff-93ae-b3da1abfe051');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 432
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# SibSp에 따른 생존율\n",
        "train_df.groupby(['SibSp'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "R3qjQFq6zHOM",
        "outputId": "5b2d4685-1798-4b70-b947-7fe409b6820b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc117b8a220>"
            ]
          },
          "metadata": {},
          "execution_count": 433
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEGCAYAAAB1iW6ZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXRV5dn+8e+dmSGAQEAgQIAEkMmBCKgIgorgAFpHWq1afUErWFt9W7X9WUtfbWtttRUccK44o1QcEGxBJmUIqMxDmIMMYR5Dpuf3Rw6ulAYzneQ5w/VZi7Vyztnn7AsWXm72Pvu5zTmHiIiEvxjfAUREJDhU6CIiEUKFLiISIVToIiIRQoUuIhIh4nztuGnTpi4tLc3X7kVEwtKiRYt2OedSynrNW6GnpaWRlZXla/ciImHJzDad7DWdchERiRAqdBGRCKFCFxGJECp0EZEIoUIXEYkQKnQRkQihQhcRiRAq9CDZfegYb8zfTF5Bke8oIhKlVOhB8tvJy3lw0lKueGoOy7bu9x1HRKKQCj0IluTs46Ml2xjS7VQO5BVw1dNzGTcjm6JiDQ8RkdqjQq8m5xx/nLKKJvUSeOyaHky9px+Dup7Kn6eu5rrnvmTT7sO+I4pIlFChV9PMNbl8sW43owemk5wUT6O6CYwdfiZ/u+EM1uw4yJC/zebNBZvRqD8RqWkq9GooLi45Om/TuC4/7N32u+fNjGFntGLqPf04s00jHnh/Kbe/mkXuwWMe04pIpFOhV8M/v97Kqu0Hue+STiTE/fcfZctGdXjtJ7156PIuzMnexeAnZzF1+XYPSUUkGqjQqyivoIi/TFtD91YNubx7i5NuFxNj/KRvOz4a3ZdTGyYx8rVF/HLiNxzMK6jFtCISDVToVTRh3ia27jvKA0M6ExNj5W6f0TyZST89j1ED0pm4KIchf5vNgg17aiGpiEQLFXoV7D9awNgZ2fTrmMK56U0r/L6EuBjuu6QT795xDrExxvXjv+QPU1ZyrFA3I4lI9anQq+DZmevYf7SA+wd3rtL7e7ZtzCd3n88NZ7fhuZnrGTZ2Lqu2HwhyShGJNhUqdDMbbGarzSzbzO4v4/VbzCzXzL4O/Lo9+FFDw7b9R3lpzgauPKMVXVo2qPLn1EuM4w8/6M6LN2ey69Axhj41l+dnradYNyOJSBWVW+hmFguMA4YAXYDhZtaljE3fds6dEfj1QpBzhownPluDc/CLizsG5fMuPK05U+/pxwWdUnjkk5UMf34eOXuPBOWzRSS6VOQIvReQ7Zxb75zLB94ChtVsrNC0ZsdBJi7K4cfntKV147pB+9wm9RN57qae/PmaHiz/9gBDnpzNe4tydDOSiFRKRQq9FbCl1OOcwHMnutrMlpjZRDNrXdYHmdkIM8sys6zc3NwqxPXrsU9XUS8xjrsGpAf9s82MazNbM+Vn53Naiwbc++433DlhMXsO5wd9XyISmYJ1UfRDIM051wP4DHi1rI2cc+Odc5nOucyUlJQg7bp2LNiwh3+t3MmdF3TglHoJNbaf1o3r8uaIPjwwpDPTV+1k0BOzmLFqZ43tT0QiR0UKfStQ+og7NfDcd5xzu51zx+9rfwHoGZx4ocE5xx+mrOTUBkncem67Gt9fbIwxsn8HPhh1Hk3rJ3DrKwv59aSlHMkvrPF9i0j4qkihLwQyzKydmSUANwCTS29gZqVvlRwKrAxeRP+mLt/OV5v38fOLM6iTEFtr+z2tRQM+GHUeI/u1540Fm7n0b7NZvHlvre1fRMJLuYXunCsERgFTKSnqd5xzy81sjJkNDWx2t5ktN7NvgLuBW2oqcG0rLCrmsU9Xk9GsPleflVrr+0+Mi+WBS0/jzf/pQ0GR45pnvuCv01ZTUFRc61lEJLSZr29SZGZmuqysLC/7rozX52/i15OW8fyPM7m4S3OvWQ7mFfC7D1cwcVEO3Vs15InrTye9WbLXTCJSu8xskXMus6zXdKfo9ziSX8iT/1rL2WmncNFpzXzHITkpnsevPZ1nbzyLnL1HuOzvc3hl7gbdjCQigAr9e704ewO5B49x/5DOmJW/AFdtGdytBVN/3o9zOzTh4Q9XcPPLC9i+P893LBHxTIV+ErsPHeO5Weu5pGtzerZt7DvOf2mWnMRLt5zNo1d1J2vjXgY9MZPJ33zrO5aIeKRCP4mnpmdztKCIX1ZxAa7aYGb8sHcbpvzsfDo0q8/db37F6De/Yv8RrbUuEo1U6GXYvPsIr8/fxHWZremQUt93nHKlNa3HuyPP4b5BHZmydBuXPDmLOWt3+Y4lIrVMhV6Gx6etJi4mhp9flOE7SoXFxcYwamAGk356HvUSY7nxxfk8PHk5eQVaa10kWqjQT7A0Zz+Tv/mW2/q2o1mDJN9xKq17akM+vvt8bjk3jVe+2Mhlf5/N0pz9vmOJSC1QoZfinOOPn66kcb0ERvZv7ztOlSXFx/Lw0K5MuK03h48VcfUzX7Au95DvWCJSw1Topcxeu4u52bsZPTCd5KR433GqrW9GUyaPOo+YGHjm83W+44hIDVOhBxQXO/44ZRWtG9fhh73b+I4TNM0aJDG8VxsmfbWVLXs0OEMkkqnQAyZ/8y0rth3gvkGdSIyrvQW4asOIfu2JMXhulo7SRSKZCh04VljE49NW061VA67o0dJ3nKBr0bAO1/RM5Z2sHHYe0B2lIpFKhQ689uUmcvYe5f7BpxETEzq3+AfTHf07UFhUzPOz1/uOIiI1JOoL/UBeAWNnZHN+RlP6ZjT1HafGtG1Sj2FntOL1+Zs11k4kQkV9oT/7+Tr2HSngVyF8i3+w/PSCDhzJL+LluRt8RxGRGhDVhb59fx4vzd3AlWe0pFurhr7j1LiM5skM7noqr3yxkQN5Wu9FJNJEdaE/+a81FBfDvYM6+Y5Sa+4akM7BvEJe+3KT7ygiEmRRW+hrdxzknawt3NinLa0b1/Udp9Z0T21I/44pvDRnA0fztc6LSCSJ2kJ/bOpq6iXEMWpguu8otW70wHR2H87nzQWbfUcRkSCKykLP2riHz1bs4I4LOtC4XoLvOLUuM60xvds15rlZ6zhWqKN0kUgRdYXunOPRT1bSvEEiPzmvne843owamM6OA8d4b9FW31FEJEiirtCnrdjB4s37+PlFHamTEFm3+FdG3/SmnJ7akGdnrqOwqNh3HBEJgqgq9MKiYh77dBUdUupxTc9U33G8MjNGDcxg854jfLhEs0hFIkFUFfq7i3JYl3uYXw3uTFxsVP3Wy3Rh52Z0PjWZcTPWUVzsfMcRkWqKmlY7kl/IE5+toWfbU7i4S3PfcUJCTIzx0wHpZO88xNTl233HEZFqippCf2nOBnYePMaDl3bGLDIX4KqKy7q3oF3TeoydkY1zOkoXCWdRUeh7Dufz7Mz1DOrSnJ5tG/uOE1JiY4w7+3dg+bcH+HxNru84IlINUVHoT01fy5H8Qn45OHpu8a+MK89sRatGdRg7XUfpIuGsQoVuZoPNbLWZZZvZ/d+z3dVm5swsM3gRq2fLniNMmLeJ689uTXqzZN9xQlJCXAwj+7dn0aa9zFu/x3ccEamicgvdzGKBccAQoAsw3My6lLFdMvAzYH6wQ1bH49NWExtj3HNRR99RQtp1ma1pWj+RcTOyfUcRkSqqyBF6LyDbObfeOZcPvAUMK2O73wN/AkJmxtmyrfv54Otvua1vO5o3SPIdJ6QlxcfyP+e3Y072Lr7ess93HBGpgooUeitgS6nHOYHnvmNmZwGtnXMff98HmdkIM8sys6zc3Jq/APenT1dxSt14RvbvUOP7igQ/6tOWRnXjGTtdR+ki4ajaF0XNLAb4K3Bveds658Y75zKdc5kpKSnV3fX3mr02l9lrdzFqYAYNkuJrdF+Ron5iHLee245/rdzBym0HfMcRkUqqSKFvBVqXepwaeO64ZKAb8LmZbQT6AJN9XhgtLnb8ccoqUk+pw4192viKEZZuOTeN+olxOpcuEoYqUugLgQwza2dmCcANwOTjLzrn9jvnmjrn0pxzacA8YKhzLqtGElfAh0u+Zfm3B7hvUCcS46J3Aa6qaFg3nhv7tOXjpdtYn3vIdxwRqYRyC905VwiMAqYCK4F3nHPLzWyMmQ2t6YCVdaywiD9PXU3Xlg0YenpL33HC0m1925EQG8Mzn6/zHUVEKiGuIhs55z4BPjnhuYdOsu0F1Y9Vda/P20zO3qP84QfdiYnRLf5VkZKcyPBebZgwbxM/uyiD1FOiZ0SfSDiLqDtFD+QV8NT0tfRNb8r5GTV70TXSjezfHjN4buZ631FEpIIiqtCfm7mOvUcKuH9IZ99Rwl6LhnW4+qxU3s7aws4DIXNrgYh8j4gp9B0H8nhxzgaGndGSbq0a+o4TEe7o34HComJemLPBdxQRqYCIKfQn/7WGomLHfYO0AFewpDWtx9DTWzJh3ib2Hs73HUdEyhERhZ698xBvL9zCjX3a0rqxLuAF008HpHMkv4iXv9joO4qIlCMiCv2xT1dRNyGOUQPSfUeJOB2bJ3NJ1+a8MncDB/MKfMcRke8R9oWetXEP01bs4I7+7WlSP9F3nIg0akAGB/IKeW3eJt9RROR7hHWhO1dyi3+z5ER+0red7zgRq3tqQ/p1TOHF2Rs4ml/kO46InERYF/pnK3aQtWkv91zUkboJFbpHSqpo9MB0dh/O562Fm31HEZGTCNtCLywq5rGpq2mfUo/rMlN9x4l4Z6c1ple7xoyftZ5jhTpKFwlFYVvoExflkL3zEL+8pDNxsWH72wgrowaks21/Hu8v3lr+xiJS68KyCY/mF/HEv9bQs+0pXNK1ue84UeP8jKb0SG3IM5+vo7Co2HccETlBWBb6S3M3sOPAMe4f0hkzLcBVW8yMuwaks3nPET5ass13HBE5QdgV+p7D+Tz7+TouOq05Z6c19h0n6lx8WnM6NU9m3Ixsioud7zgiUkrYFfpLczZwOL+QXw3WLf4+xMQYPx3QgbU7DzFtxXbfcUSklLAr9LsGpPPCzZlkNE/2HSVqXd6jJWlN6jJ2RjbO6ShdJFSEXaHXSYhlYGddCPUpNsa484IOLNt6gJlrcn3HEZGAsCt0CQ1XnZlKy4ZJGiYtEkJU6FIlCXExjOzfgYUb9zJ//W7fcUQEFbpUw/Vnt6Zp/UTG6ihdJCSo0KXKkuJjuf38dsxeu4tvtuzzHUck6qnQpVpu7NOWhnXidZQuEgJU6FIt9RPjuPW8ND5bsYNV2w/4jiMS1VToUm23nJtGvYRYnp6xzncUkaimQpdqa1Q3gRvPactHS75lw67DvuOIRC0VugTF7X3bEx8bwzOf61y6iC8qdAmKlOREhvdqw/uLt7J131HfcUSikgpdgmZEv/aYwfiZOpcu4kOFCt3MBpvZajPLNrP7y3j9DjNbamZfm9kcM+sS/KgS6lo2qsMPzkzlzYVb2Hkwz3cckahTbqGbWSwwDhgCdAGGl1HYbzjnujvnzgAeA/4a9KQSFu68oAOFRcW8OHuD7ygiUaciR+i9gGzn3HrnXD7wFjCs9AbOudJfQK4HaE3VKJXWtB6X92jJhHmb2Hck33cckahSkUJvBWwp9Tgn8Nx/MLO7zGwdJUfod5f1QWY2wsyyzCwrN1fLrkaquwakczi/iJfnbvQdRSSqBO2iqHNunHOuA/Ar4Dcn2Wa8cy7TOZeZkpISrF1LiOl0ajKDujTnlS82cjCvwHcckahRkULfCrQu9Tg18NzJvAVcWZ1QEv5GDUxn/9ECJszb7DuKSNSoSKEvBDLMrJ2ZJQA3AJNLb2BmGaUeXgasDV5ECUc9UhtxfkZTXpyznryCIt9xRKJCuYXunCsERgFTgZXAO8655WY2xsyGBjYbZWbLzexr4BfAzTWWWMLG6IEZ7DqUz1sLdJQuUhvM15DfzMxMl5WV5WXfUnuue/ZLtuw9wsz/HUBCnO5jE6kuM1vknMss6zX9FyY16q6B6Wzbn8ekr3J8RxGJeCp0qVH9MprSvVVDnv58HYVFxb7jiEQ0FbrUKDPjrgHpbNp9hI+XbvMdRySiqdClxg3q0pyOzeszbkY2xcW6iVikpqjQpcbFxJQcpa/ZcYjPVu7wHUckYqnQpVZc1r0FbZvUZez0bHx9s0ok0qnQpVbExcZwZ/8OLN26n1lrd/mOIxKRVOhSa35wViotGiYxbrrG1InUBBW61JqEuBhG9mvPgo17WLBhj+84IhFHhS616oZebWhaP4GxM3SULhJsKnSpVUnxsdzWtz2z1uTyzZZ9vuOIRBQVutS6G/u0oUFSHON0lC4SVCp0qXXJSfHcel47pq3YwevzN/mOIxIxVOjixYh+7bmgUwq/nrSMByctJb9Q67yIVJcKXbyolxjHizefzR39O/DG/M386IV55B485juWSFhToYs3sTHG/UM689TwM1m6dT9Dx85hSY4ulIpUlQpdvLvi9Ja8d+e5xJhx7bNfau10kSpSoUtI6NqyIZNHnceZbRrx87e/4f8+WqH100UqSYUuIaNJ/UReu603t5ybxgtzNnDLywvZezjfdyyRsKFCl5ASHxvDw0O78tg1PViwYQ9Dx81h1fYDvmOJhAUVuoSk6zJb89bIPhwrKOYHT3/BFE07EimXCl1C1lltTuGj0X3pdGoyd76+mL9MW62JRyLfQ4UuIa1ZgyTeGtGH6zNb89T0bEa8lsXBvALfsURCkgpdQl5iXCx/vLo7vx/Wlc9X53LluLmsyz3kO5ZIyFGhS1gwM246J40Jt/dm35ECrhw7lxmrdvqOJRJSVOgSVvq0b8Lk0X1p06QuP3l1IeNmaEapyHEqdAk7rRrVYeId53JFj5b8eepqRr3xFUfyC33HEvFOhS5hqU5CLH+74QweGNKZKcu28YOnv2DLniO+Y4l4VaFCN7PBZrbazLLN7P4yXv+Fma0wsyVm9m8zaxv8qCL/ycwY2b8DL9/ai2/3HWXo2Dl8kb3LdywRb8otdDOLBcYBQ4AuwHAz63LCZl8Bmc65HsBE4LFgBxU5mf4dU5g8qi9N6ydy00sLeGnOBp1Xl6hUkSP0XkC2c269cy4feAsYVnoD59wM59zxf+/OA1KDG1Pk+6U1rceku87jws7NGPPRCu57dwl5BUW+Y4nUqooUeitgS6nHOYHnTuY2YEpZL5jZCDPLMrOs3NzciqcUqYD6iXE8e2NP7rkog/cW53D9+Hls35/nO5ZIrQnqRVEzuxHIBP5c1uvOufHOuUznXGZKSkowdy0CQEyMcc9FHRl/U0+ydxzk8qfmsGjTHt+xRGpFRQp9K9C61OPUwHP/wcwuAn4NDHXOaZaYeDWo66lMuus86ifGcsP4eby5YLPvSCI1riKFvhDIMLN2ZpYA3ABMLr2BmZ0JPEdJmev2PQkJHZsn88FdfTmnQ1MeeH8pv/mnhlFLZCu30J1zhcAoYCqwEnjHObfczMaY2dDAZn8G6gPvmtnXZjb5JB8nUqsa1o3n5VvOZmT/9kyYt5kbX5jPrkP6B6REJvP19a7MzEyXlZXlZd8SnT74eiu/em8Jjesm8NxNmXRPbeg7kkilmdki51xmWa/pTlGJGsPOaMXEO87FzLjm2S/451f/dSlIJKyp0CWqdGtVMoz6jNaNuOftr3nkYw2jlsihQpeo06R+IhNu783N57Tl+dkbuPWVhew7omHUEv5U6BKV4mNj+N2wbjx2dQ/mr9/D0LFzWb39oO9YItWiQpeodt3ZrXlzRB/yCoq46um5fLpMw6glfKnQJer1bHsKH47uS8fmydwxYTF//WyNFveSsKRCFwGaB4ZRX9szlb//ey1vLthS/ptEQowKXSQgKT6Wx67pQe92jfnTp6t0A5KEHRW6SClmxiNXdedIfiGPfrzSdxyRSlGhi5wgvVl97ujfgfe/2soX6zQBScKHCl2kDHcNSKdN47r8ZtIyjhVqUIaEBxW6SBmS4mP5/ZXdWL/rMM/NXO87jkiFqNBFTqJ/xxQu79GCsTOy2bDrsO84IuVSoYt8j4cu70JibAwPfbBM302XkKdCF/kezRok8b+DOzF77S4+XKK7SCW0qdBFyvGj3m3pkdqQMR+uYP/RAt9xRE5KhS5SjtgY49GrurPn8DEen7radxyRk1Khi1RAt1YNufncNCbM38TXW/b5jiNSJhW6SAX94uKONEtO5MH3l2oohoQkFbpIBSUnxfPwFV1Zse0Ar365yXcckf+iQhephMHdTmVApxT+Om012/Yf9R1H5D+o0EUqwcwYM6wbRc7xu8krfMcR+Q8qdJFKat24LndfmMGny7fz75U7fMcR+Y4KXaQKbu/bnoxm9Xnog+UcyS/0HUcEUKGLVElCXAyPXNWdrfuO8rd/r/UdRwRQoYtUWa92jbkuM5UXZ29g1fYDvuOIqNBFquOBIaeRnBTHbyYto7hYi3eJXyp0kWo4pV4CD156Glmb9vJOlgZLi18VKnQzG2xmq80s28zuL+P1fma22MwKzeya4McUCV3X9EylV7vG/GHKKnZrsLR4VG6hm1ksMA4YAnQBhptZlxM22wzcArwR7IAioc7MePSqbhzJL+SRTzRYWvypyBF6LyDbObfeOZcPvAUMK72Bc26jc24JoAUuJCqlN0tmRL/2vL9Yg6XFn4oUeiug9MnBnMBzlWZmI8wsy8yycnNzq/IRIiFr9MCMksHS/9RgafGjVi+KOufGO+cynXOZKSkptblrkRqXFB/LmGFdWZ97mPEaLC0eVKTQtwKtSz1ODTwnIie4oFMzLuvRgqdmZLNRg6WlllWk0BcCGWbWzswSgBuAyTUbSyR8PXR5FxJiY/h/GiwttazcQnfOFQKjgKnASuAd59xyMxtjZkMBzOxsM8sBrgWeM7PlNRlaJJQ1b5DEfYM6MnvtLj7SYGmpRebrCCIzM9NlZWV52bdITSsqdlw5bi7bD+Tx73v70yAp3nckiRBmtsg5l1nWa7pTVKQGHB8svfuQBktL7VGhi9SQ7qkN+fE5abw2bxPfaLC01AIVukgNundQR1LqJ/LgJA2WlpqnQhepQclJ8fz2iq4s//YA/9BgaalhKnSRGnZp91O5oFMKf9FgaalhKnSRGmZmjBnajcJix5gPNVhaao4KXaQWtGlSMlh6yrLtTF+lwdJSM1ToIrXkf85vT3pgsPTRfC3eJcGnQhepJQlxMTxyZTdy9h7l79M1WFqCT4UuUot6t2/CtT1TeX7WelZvP+g7jkQYFbpILXvg0sBg6X8u1WBpCSoVukgta1wvgQcuPY2FG/cycVGO7zgSQVToIh5c2zOVXmmNeXTKSg2WlqBRoYt4YGY8clU3DuUV8ocpq3zHkQihQhfxJKN5yWDpiYtymLd+t+84EgFU6CIejR6YQevGdfj1pKXkF2rxLqkeFbqIR3USYhkztBvrcg8zftY633EkzKnQRTwb0LkZl3Y/laemZ7NptwZLS9Wp0EVCwEOXdyU+NoaHPliuwdJSZSp0kRBwasMk7h3UkZlrcvl4qQZLS9Wo0EVCxI/PSaNbqwaM+XAFB/IKfMeRMKRCFwkRxwdL5x46xl80WFqqQIUuEkJ6pDbix33a8o95m1iSo8HSUjkqdJEQc+8lnb4bLF2kxbukElToIiGmQVI8D13RhWVbD/CPLzf6jiNhRIUuEoIu696C/h1T+Mu0NWzfn+c7joQJFbpICDIzxgzrSkFRMWM+Wu47joQJFbpIiGrbpB6jB6bzydLtzFi103ccCQMVKnQzG2xmq80s28zuL+P1RDN7O/D6fDNLC3ZQkWg0ol+HksHSk5dpsLSUq9xCN7NYYBwwBOgCDDezLidsdhuw1zmXDjwB/CnYQUWiUUJcDP93ZTe27DnKUxosLeWIq8A2vYBs59x6ADN7CxgGrCi1zTDg4cDPE4GxZmZOi1KIVFuf9k24pmcqz81az2crdviOI0Fw94UZXHF6y6B/bkUKvRWwpdTjHKD3ybZxzhWa2X6gCbCr9EZmNgIYAdCmTZsqRhaJPr++9DQS4mLYdyTfdxQJgoZ14mvkcytS6EHjnBsPjAfIzMzU0btIBZ1SL4FHr+ruO4aEuIpcFN0KtC71ODXwXJnbmFkc0BDQTC0RkVpUkUJfCGSYWTszSwBuACafsM1k4ObAz9cA03X+XESkdpV7yiVwTnwUMBWIBV5yzi03szFAlnNuMvAi8JqZZQN7KCl9ERGpRRU6h+6c+wT45ITnHir1cx5wbXCjiYhIZehOURGRCKFCFxGJECp0EZEIoUIXEYkQ5uvbhWaWC2yq4tubcsJdqCFCuSpHuSovVLMpV+VUJ1db51xKWS94K/TqMLMs51ym7xwnUq7KUa7KC9VsylU5NZVLp1xERCKECl1EJEKEa6GP9x3gJJSrcpSr8kI1m3JVTo3kCstz6CIi8t/C9QhdREROoEIXEYkQYVfo5Q2s9sHMXjKznWa2zHeW0systZnNMLMVZrbczH7mOxOAmSWZ2QIz+yaQ63e+M5VmZrFm9pWZfeQ7y3FmttHMlprZ12aW5TvPcWbWyMwmmtkqM1tpZueEQKZOgT+n478OmNk9vnMBmNnPA3/nl5nZm2aWFNTPD6dz6IGB1WuAiykZhbcQGO6cW/G9b6z5XP2AQ8A/nHPdfGYpzcxaAC2cc4vNLBlYBFwZAn9eBtRzzh0ys3hgDvAz59w8n7mOM7NfAJlAA+fc5b7zQEmhA5nOuZC6ScbMXgVmO+deCMxLqOuc2+c713GBztgK9HbOVfVGxmBlaUXJ3/UuzrmjZvYO8Ilz7pVg7SPcjtC/G1jtnMsHjg+s9so5N4uSdeBDinNum3NuceDng8BKSua/euVKHAo8jA/8CokjCzNLBS4DXvCdJdSZWUOgHyXzEHDO5YdSmQdcCKzzXealxAF1ApPd6gLfBvPDw63QyxpY7b2gwoGZpQFnAvP9JikROK3xNbAT+Mw5FxK5gCeBXwLFvoOcwAHTzGxRYNh6KGgH5AIvB05RvWBm9XyHOsENwJu+QwA457YCjwObgW3AfufctGDuI9wKXarAzOoD7wH3OOcO+M4D4Jwrcs6dQcmM2l5m5v1UlZldDux0zi3ynaUMfZ1zZwFDgLsCp/l8iwPOAp5xzp0JHAZC4roWQOAU0FDgXd9ZAMzsFJ7U4ZsAAANUSURBVErOKLQDWgL1zOzGYO4j3Aq9IgOrpZTAOer3gNedc+/7znOiwD/RZwCDfWcBzgOGBs5XvwUMNLMJfiOVCBzd4ZzbCUyi5PSjbzlATql/XU2kpOBDxRBgsXNuh+8gARcBG5xzuc65AuB94Nxg7iDcCr0iA6slIHDx8UVgpXPur77zHGdmKWbWKPBzHUoucq/ymwqccw8451Kdc2mU/N2a7pwL6hFUVZhZvcBFbQKnNAYB3r9R5ZzbDmwxs06Bpy4EvF5wP8FwQuR0S8BmoI+Z1Q38t3khJde1gqZCM0VDxckGVnuOhZm9CVwANDWzHOC3zrkX/aYCSo44bwKWBs5XAzwYmBHrUwvg1cA3EGKAd5xzIfMVwRDUHJhU0gHEAW845z71G+k7o4HXAwdY64FbPecBvvsf38XASN9ZjnPOzTezicBioBD4iiAvARBWX1sUEZGTC7dTLiIichIqdBGRCKFCFxGJECp0EZEIoUIXEYkQKnSJeGb268AKd0sCq+/1Dtym3iXw+qGTvK+Pmc0PvGelmT1cq8FFKimsvocuUlmB5VwvB85yzh0zs6ZAgnPu9gq8/VXgOufcN4HvzHcq7w0iPukIXSJdC2CXc+4YgHNul3PuWzP73Mwyj29kZk8EjuL/bWYpgaebUbKI0vG1Z1YEtn3YzF4zsy/NbK2Z/U8t/55EyqRCl0g3DWhtZmvM7Gkz61/GNvWALOdcV2Am8NvA808Aq81skpmNPGEYQQ9gIHAO8JCZtazB34NIhajQJaIF1l3vCYygZKnXt83slhM2KwbeDvw8AegbeO8YSgZdTAN+CJS+3f4D59zRwMCJGYTGYlkS5XQOXSKec64I+Bz43MyWAjeX95ZS710HPGNmzwO5ZtbkxG1O8lik1ukIXSJaYL5kRqmnzgBOnF4TA1wT+PmHlIwJw8wuC6yKB5ABFAHHJ/IMs5LZqE0oWZhtYQ3EF6kUHaFLpKsPPBVYrrcQyKbk9MvEUtscpmTIxm8omaB0feD5m4AnzOxI4L0/cs4VBTp+CSWnWpoCv3fOBXWUmEhVaLVFkUoKfB/9kHPucd9ZRErTKRcRkQihI3QRkQihI3QRkQihQhcRiRAqdBGRCKFCFxGJECp0EZEI8f8BNQ3qZ9dfLOkAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df[[\"SibSp\", \"Survived\"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "cUhsztK7zTEh",
        "outputId": "dcca7345-8d95-4b39-c48b-8eb29e324bb3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   SibSp  Survived\n",
              "1      1  0.535885\n",
              "2      2  0.464286\n",
              "0      0  0.345395\n",
              "3      3  0.250000\n",
              "4      4  0.166667\n",
              "5      5  0.000000\n",
              "6      8  0.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b161e985-6705-4ac5-b2e3-ef442f03f035\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.535885</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>0.464286</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.345395</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>0.250000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>0.166667</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>5</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>8</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b161e985-6705-4ac5-b2e3-ef442f03f035')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b161e985-6705-4ac5-b2e3-ef442f03f035 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b161e985-6705-4ac5-b2e3-ef442f03f035');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 434
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Parch에 따른 생존율\n",
        "train_df.groupby(['Parch'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "dfTfAKc6zWu6",
        "outputId": "a94924ad-1afe-4c4b-e7e3-928419482371"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc117b43280>"
            ]
          },
          "metadata": {},
          "execution_count": 435
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df[[\"Parch\", \"Survived\"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "BcMqDZrnzdw7",
        "outputId": "81ec0391-4652-4d8d-d777-0592b46a645a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Parch  Survived\n",
              "3      3  0.600000\n",
              "1      1  0.550847\n",
              "2      2  0.500000\n",
              "0      0  0.343658\n",
              "5      5  0.200000\n",
              "4      4  0.000000\n",
              "6      6  0.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-025829c1-fa00-415c-aaf9-46605c11b409\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Parch</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>0.600000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.550847</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>0.500000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.343658</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>5</td>\n",
              "      <td>0.200000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>6</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-025829c1-fa00-415c-aaf9-46605c11b409')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-025829c1-fa00-415c-aaf9-46605c11b409 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-025829c1-fa00-415c-aaf9-46605c11b409');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 436
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Fare에 따른 생존율 \n",
        "train_df[[\"Fare\", \"Survived\"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "hQiMTPUV7e1z",
        "outputId": "331d889c-f553-4d24-9ff6-c261f12c5be5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         Fare  Survived\n",
              "247  512.3292       1.0\n",
              "196   57.9792       1.0\n",
              "89    13.8583       1.0\n",
              "88    13.7917       1.0\n",
              "86    13.4167       1.0\n",
              "..        ...       ...\n",
              "103   15.5500       0.0\n",
              "180   47.1000       0.0\n",
              "179   46.9000       0.0\n",
              "178   42.4000       0.0\n",
              "124   21.0750       0.0\n",
              "\n",
              "[248 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9664b32d-0a4e-4451-8fe5-17c0e809b2bf\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Fare</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>247</th>\n",
              "      <td>512.3292</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>196</th>\n",
              "      <td>57.9792</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>89</th>\n",
              "      <td>13.8583</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>88</th>\n",
              "      <td>13.7917</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>86</th>\n",
              "      <td>13.4167</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>103</th>\n",
              "      <td>15.5500</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>180</th>\n",
              "      <td>47.1000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>179</th>\n",
              "      <td>46.9000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>178</th>\n",
              "      <td>42.4000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>124</th>\n",
              "      <td>21.0750</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>248 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9664b32d-0a4e-4451-8fe5-17c0e809b2bf')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9664b32d-0a4e-4451-8fe5-17c0e809b2bf button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9664b32d-0a4e-4451-8fe5-17c0e809b2bf');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 437
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Age에 따른 생존률\n",
        "train_df[[\"Age\", \"Survived\"]].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "LYL4QOgpAVfK",
        "outputId": "b28be102-a089-46ea-fe4e-1946bcd64e7f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Age  Survived\n",
              "0  0.0  0.579710\n",
              "1  1.0  0.428571\n",
              "3  3.0  0.412698\n",
              "2  2.0  0.385230\n",
              "4  4.0  0.090909"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-cc5ae10d-128d-4830-9ef2-11481595d4f3\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.579710</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1.0</td>\n",
              "      <td>0.428571</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3.0</td>\n",
              "      <td>0.412698</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2.0</td>\n",
              "      <td>0.385230</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4.0</td>\n",
              "      <td>0.090909</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cc5ae10d-128d-4830-9ef2-11481595d4f3')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-cc5ae10d-128d-4830-9ef2-11481595d4f3 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-cc5ae10d-128d-4830-9ef2-11481595d4f3');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 438
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.groupby(['Age'])['Survived'].mean().plot()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "3uIME1bKAdlC",
        "outputId": "6ff29fbd-106c-47ca-a140-0043ee09840c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc10ebe2760>"
            ]
          },
          "metadata": {},
          "execution_count": 439
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">### **2.3 데이터 시각화**\n",
        ">> 열들을 시각화 하여 대략적인 분포를 파악하고, 생존과의 상관관계를 유추한다. 명목형, 이산형 데이터에 대해 막대 그래프를 그려준다. "
      ],
      "metadata": {
        "id": "VoJF95b7-O2u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def bar_chart(feature):\n",
        "  # 각 column(=feature)에서 생존자의 수 count\n",
        "  survived = train_df[train_df['Survived']==1][feature].value_counts()\n",
        "\n",
        "  # 각 column(=feature)에서 사망자의 수 count\n",
        "  dead = train_df[train_df['Survived']==0][feature].value_counts()\n",
        "\n",
        "  # 생존자의 수, 사망자의 수를 하나로 묶기\n",
        "  df= pd.DataFrame([survived,dead])\n",
        "\n",
        "  # 묶은 dataframe의 인덱스명(행이름) 지정\n",
        "  df.index=['Survived','Dead']\n",
        "\n",
        "  # plot 그리기\n",
        "  df.plot(kind='bar', stacked=True, figsize=(10,5))"
      ],
      "metadata": {
        "id": "Qc1pwiCI_0hw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 성별 차트\n",
        "bar_chart('Sex') # 남성이 여성에 비해 더 많이 사망한 것을 알 수 있다"
      ],
      "metadata": {
        "id": "UHG0OAZc9LkB",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "outputId": "641dc61c-d4f6-42f4-e3e8-3ca79597e61b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Age\n",
        "\n",
        "bar_chart('Age')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "id": "gabWlgMgAmfh",
        "outputId": "4d34827b-a64a-448b-b50b-91d0f85c32d3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 훈련 데이터의 Sex별 생존자 수치 분석\n",
        "train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)\n",
        "\n",
        "# 다른 팀원분이 한거 보니까 Sex 부분에 female, male 부분이 적혀져서 나오던데, 이건 어떻게 하는건지...!"
      ],
      "metadata": {
        "id": "mdl0N5aGrKOc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        },
        "outputId": "7e31ee0a-db11-4345-a0f4-11772dc659c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Sex  Survived\n",
              "0    0  0.742038\n",
              "1    1  0.188908"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5415cdf7-74e2-4b73-bfe7-c665d7aeaee8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.742038</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.188908</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5415cdf7-74e2-4b73-bfe7-c665d7aeaee8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-5415cdf7-74e2-4b73-bfe7-c665d7aeaee8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5415cdf7-74e2-4b73-bfe7-c665d7aeaee8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 443
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### * as_index = False: 구문은 이 그룹을 인덱스로 지정할 것인지 여부인데, 인덱스로 지정하면 그룹이 표의 형태로 들어간다\n",
        "##### -> 위처럼 표로 나타내어지는 이유가 as_index = False 때문이다\n",
        "\n",
        "##### * ascending = False : 내림차순 정렬을 의미한다\n",
        "##### * 이 외의 정렬에는\n",
        ">##### 1) Row index 기준정렬\n",
        ">##### : Series 객체 index 순으로 정렬한다\n",
        ">##### s1.sort_index()\n",
        ">##### : DataFrame 객제 index 순으로 결정\n",
        ">##### df2.sort_index()\n",
        "\n",
        ">##### 2) Column index 기준정렬\n",
        ">##### : DataFrame 객체 column 순으로 정렬하여 axis를 지정해야한다.\n",
        ">##### df2.sort_index(axis=1)\n",
        "\n",
        ">##### 3) 객체를 값에 따라 정렬한다\n",
        ">##### : 객체를 값에 따라 정렬할 경우에는 sort_values 메스드를 이용하고, 기본적으로 오름차순으로 정렬된다\n"
      ],
      "metadata": {
        "id": "QjkAPQ85sC0d"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 사회, 경제적 지위별 차트\n",
        "bar_chart('Pclass') # 3st에 해당하는 사람들이 훨씬 더 많이 죽은 것을 확인 할 수 있다"
      ],
      "metadata": {
        "id": "0majK_JWCK-b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "outputId": "583ded1c-b9e6-4118-c116-9fc8faf2c9ba"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 훈련 데이터의 Pclass별 생존자 수치 분석\n",
        "train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived' )"
      ],
      "metadata": {
        "id": "dW08IS-lqfk6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "outputId": "f5b3e509-1396-467f-ba87-f2ab534eb78f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Pclass  Survived\n",
              "2       3  0.242363\n",
              "1       2  0.472826\n",
              "0       1  0.629630"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5405e595-3c9f-47b1-90c1-750268dcb07b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>0.242363</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>0.472826</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0.629630</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5405e595-3c9f-47b1-90c1-750268dcb07b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-5405e595-3c9f-47b1-90c1-750268dcb07b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5405e595-3c9f-47b1-90c1-750268dcb07b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 445
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 탑승지역(항구위치)별 차트\n",
        "bar_chart('Embarked') # Southampton에서 탑승한 사람들이 훨씬 더 많이 생존하고, 사망한 것을 알 수 있다"
      ],
      "metadata": {
        "id": "bWinJWavC334",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "outputId": "efad8744-2871-436d-93d4-6394e6646b32"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Embarked \n",
        "##### train 부분에 NaN 값이 2개가 있고, 이를 처리를 안해주면 나중에 오류가 날 수 있기 때문에, 최빈값인 S로 일괄 채워줘야 합니다.\n",
        "\n",
        "##### 또한, Embarked 칼럽 역시 numberic한 데이터로 변경해서 'Embarked_clean' 칼럼에 채워줘야 합니다."
      ],
      "metadata": {
        "id": "dkRrwqqOFK3k"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df['Embarked'].isnull().sum()\n",
        "test_df['Embarked'].isnull().sum()\n",
        "\n",
        "train_df['Embarked'].value_counts()\n",
        "\n",
        "train_df['Embarked'].fillna('S', inplace=True)"
      ],
      "metadata": {
        "id": "YrN8A-fTFKX8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 타이타닉호에 탑승한 형제자매 수 별 차트\n",
        "bar_chart('SibSp') # 0명인 사람이 많이 죽기도 하였지만, 상대적으로 많이 생존한 것을 알 수 있다"
      ],
      "metadata": {
        "id": "jcWFTAwADaKX",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "outputId": "c01f86d2-f33a-47c8-c6fb-5fdb0b46170c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">##### 남성보단 여성이 많이 생존하였고, 1등급 승객, 가족이 있는 승객의 생존율이 더욱 높은 것을 알 수 있다. 탑승지역에서는 S 승객들이 많이 사망한 것으로 보인다"
      ],
      "metadata": {
        "id": "mdjRCRSqEJ2Y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#train 데이터의 SipSp 별 생존자 수치 분석\n",
        "train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived' )"
      ],
      "metadata": {
        "id": "0woI8B7Oab_o",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "outputId": "7a726ef0-70b1-4f18-b5af-4951305b1508"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   SibSp  Survived\n",
              "5      5  0.000000\n",
              "6      8  0.000000\n",
              "4      4  0.166667\n",
              "3      3  0.250000\n",
              "0      0  0.345395\n",
              "2      2  0.464286\n",
              "1      1  0.535885"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9c276f4d-cfda-44eb-9641-d269b90a6468\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>5</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>8</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>0.166667</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>0.250000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.345395</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>0.464286</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.535885</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9c276f4d-cfda-44eb-9641-d269b90a6468')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9c276f4d-cfda-44eb-9641-d269b90a6468 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9c276f4d-cfda-44eb-9641-d269b90a6468');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 449
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Age(나이) \n",
        "bar_chart('Age')\n",
        "\n",
        "# 범위를 지정해서 소트 시키기 -> 학습진행"
      ],
      "metadata": {
        "id": "9Y0te3t4Dy7q",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "outputId": "048b682a-861e-4997-9418-f6991d7d270d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">##### 돌려보고 이렇게 구하는게 아님을 알았다..\n",
        ">##### 다양한 분포가 존재하니 막대 그래프 말고 점으로 표시된 그래프로 나타내보기로 했다"
      ],
      "metadata": {
        "id": "4f6Af1zKx-16"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "bar_chart('Fare')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "yKeP4uYa71W2",
        "outputId": "9f3a943e-ca07-4361-e64a-9e9e9365be97"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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SFpUVBQZGBjofvPNN6+N2bt3b8j06dN7b/TQOwAAERHL4/GM9hoAgGGMRmODTqfrGO11jBVGozFcp9NJRnsdAHDrcCsQAAAAwEcQrAAAAAB8BMEKAAAAwEcQrAAAAAB8BMEKAAAAwEcQrAAAAAB8BMEKABiroKBAJJPJNHK5XKPX66V2u501smbXrl2hiYmJGplMptHr9ddeacPhcFKVSqVaqVSq09PTZSPHLV++fEJgYOAk788bNmyITExM1CgUCvU999yjuHDhgv/PtzMAYCocEAoAt+ycUpXqy/lU58+duVlNfX29386dOyNNJlONQCDwZGZmJuzatSssNzf3qremurqa9+qrr0Z//fXX5yMiIlytra3XvvN4PJ77/PnztTea+/PPPw/s7u6+7vsxNTXVvmbNmnNCodD9n//5nxGrV6+OKy0tvXQr+wSAsQdXrACAsVwuF8tms7GdTic5HA52XFycc3j/66+/HrFixYp/RkREuIiIYmNjh24259DQED333HNxr732Wsvwdr1ebxUKhW4iomnTpvW1tbXhihUAfA+CFQAwklQqdebk5FikUqlWJBLphEKhy2Aw9A6vMZvNvAsXLvBTUlKUOp1OeeDAgSBv3+DgIDs5OVml0+mU77zzToi3/eWXXxZlZmZ2x8fHXxfShvvzn/8ccf/99/f8PDsDACbDrUAAYKT29nZOaWlpiNlsrh4/frxr7ty5CUVFRWErV67s9Na4XC7WxYsXeV999ZWpvr7e77777lPed999Z8PDw111dXVVUqnUWVtb6z9nzpyklJQUx7hx49wHDx4M/frrr00/9LlFRUVhRqMx8M9//vMP1gDAnQtXrACAkUpKSoLEYvFATEzMEI/H82RlZXVXVFQIhtdER0cPzps3r5vH43mUSuWgVCrtP3v2LI/ouyteRERqtXpwypQp1pMnTwZ+/fXXgY2NjXyJRHJXbGzsXf39/WyxWJzsne/gwYPCP/zhD9GffPKJOSAgAC9aBYDvQbACAEaSSCSDlZWVAqvVyna73VReXi5UqVT9w2sMBkP3sWPHhEREbW1t3Pr6en5SUtJAe3s7x+FwsLztp0+fFmi1WsdDDz3U09HRYWxtba1ubW2t5vP57qamphoioi+//DLg2Wefjf/444/NP+ZZLQC4M+FWIAAwUnp6uk2v13dptVoVl8sljUZjz8/P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lixZ0nru3LmKM2fOVHl7e/dJJBI9EVFoaKiZzWYTi8WiRYsWtZw9e9aNiGjXrl1DH3zwwS5nZ2drcHCwecSIEZcLCgrc+vfp4+PTd//992sPHjzo+UfNFwD+3BCsAMCh7dmzh5eRkXFVgPryyy89wsPD9REREaYb1TU2NrKJiGpqapwOHTrkNX/+/HYiIrVazenXt5dUKu0lIuLz+cbjx497EBF1d3czi4uL3aKjo/WXLl1it7a2soiILl++zDh+/LhHZGSk3v4zBQBHgF8FAoDD6u7uZubn53vs2LFD3f/47t27eenp6VeFrbq6Os7s2bMF33//fS0R0dSpUyM6OzvZbDbbumbNmos+Pj59RESLFy8OKS8vH0JEFBISYty+fbuaiOill176ZcaMGUKRSKSwWq2UmZnZOnLkyN5///vfQ+bMmRPW19dHVquV8dBDD7U/+uijXXfmEwCAPxuG1Wq922MAAAejUqnqlEpl690ex2ChUql8lEql8G6PAwBuHy4FAgAAANgJghUAAACAnSBYAQAAANgJghUAAACAnSBYAQAAANgJghUAAACAnSBYAYBDUqlUzjKZTG775+7uHpudne1XUFAwRKlUymQymTwqKiry+PHjrgNrCwoKhgwbNkwmEokUEolEvmXLFm/bOYvFQs8++2ywUCiMCg8PV7z++ut+tnNfffUVVyaTyUUikWLEiBHS3xrHnfkUAODPBvtYAcAtG7iP1fsLjsXbs/+FGxOLbqW92WymgIAAZUFBQcXcuXMFixcvbs7IyOjOycnxXL16dUBhYWFV//YlJSXODAaDoqOjDXV1dZwRI0ZEVlRUnPPx8elbu3bt0H/961/c3NzcOhaLRY2Njezg4GBza2sra+TIkbLDhw/XiMVio+34jcYhkUiMv3f82McKYPDAzusA4PAOHDjgwefzDRKJxMhgMKirq4tFRNTZ2cny9/e/JuDExMQYbH8LhUITj8czNzU1sX18fPq2bt3qt3v37vO2hzLbwtPWrVt5ycnJHWKx2Nj/+I3G8QdNFQD+5HApEAAc3u7du3lpaWltRETr1q2rf+WVV0ICAgJili9fHrJ69erG36o9fvy4q8lkYsjlcgMRUX19vfOuXbu8o6KiIseNGycuLS11JiKqrq526ejoYN93331ShUIR+d577w39rXEAwL0JwQoAHJper2ccPXrU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o0ePEh8fz4MPPshVVzXvfvizn/2M5cuXExMTQ1JSEs888wx9+/Zl0aJFzJo1C+cciYmJPPHEE1x66aUAPPLII5SVleGc45ZbbuHHP/4xAIWFhaxYsYL4+HgGDRrEr3/9a84999yO+wW1YvmhY15HkIBQN8VTp8LKx75R8YzXESQoPO6OXFJS4tnrVVZWhht9AgwdOpSlS5dy6623NtmvV69erFixgr59+7J161a+853vtLjAcmFhIffddx8Ajz76KKWlpcydO5eBAweyZs0azjvvPFatWkVBQQEbN25k69atlJWVUVVVRXx8POPGjeO73/0uF154Ienp6cyYMYO4uDiKioqYMWMGs2bNOr1fiogEggorH/vo9wVeR5CgmNn5Oq+3JiUlpcXtw4cPD9+++OKLOXz4MEeOHKFr165N9uvRo0f4dn19PWYGwLe+9a3w9ssvv5za2loAqqurSUtLo1u3bgCMGTOGpUuXctddd5GRkdHkOYsXLz7Do4sMreogbafPllOlwsrHJv5U/3mkbd70OoBHzIyMjAzMjFtvvZWCgrZ9GVmyZAkjRoxoVlQdN23aNBYuXMg555xDZWVls8fnz5/PNddcAzSOkE2bNo26ujrOOussVq5c2WRh5+OefvppbrjhhlM4uvajzxZpq8762XIm9K/Lx97ctdvrCCK+tm7dOvr168f+/ftJT09nyJAhXHnllV/6nG3btlFUVERFRUWr+0yfPp3p06czY8YMZs+ezc9//vPwY5WVlcyfP59169YBjSNkRUVFZGRk0L17d4YNG0ZsbGyz14uLiyM3N/cMjlZEgkCFlY+p3YK0VY3XATzSr18/AJKSksjMzKSqqupLC6va2loyMzNZuHAhgwYNOunr5+bmcu2114YLqzfeeIPJkyezatUqevbsGd4vPz8/vFjz3XffTXJycvixZ555hj/84Q/85S9/CZ9WFJHopcJKRAKpvr6ezz77jMTEROrr66moqODee+9tdf9Dhw4xfvx4Zs6cyejRo1vdb8eOHQwePBiA5cuXM2TIEAB2795NVlYW5eXlXHTRRU2es3//fpKSkti9ezdLly5lw4YNAKxevZoHHniANWvWhOdg+YFGw0XajworEQmkffv2kZmZCUBDQwM33ngj48aNY9myZfzwhz/kwIEDjB8/nmHDhvHSSy8xe/Zsdu7cSWlpKaWl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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "너무 다양한 수로 분리 되어 표가 깔끔하게 나오지 않음으로 Fare 또한 범주형으로 분리하려고 한다. \n",
        "\n",
        "* 1등실 중 응접실이 딸린 스위트 룸: 870유로, $4,350\n",
        "* 1등실중 침실칸 : 30유로, $150\n",
        "* 2등실: 12유로, $60\n",
        "* 3등실: 3~8유로, $15~40\n",
        "\n",
        "따라서 ~8유로, ~40$ 까지를 2\n",
        "~12유로 ~ 60$ 까지를 1\n",
        "그 이상은 전부 0으로 처리하겠다"
      ],
      "metadata": {
        "id": "qCThnFhR8LJB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for dataset in train_and_test:\n",
        "    dataset.loc[dataset['Fare'] <= 39, 'Fare'] = 2\n",
        "    dataset.loc[(dataset['Fare'] > 39) & (dataset['Fare'] <= 60), 'Fare'] = 1\n",
        "    dataset.loc[ dataset['Fare'] > 60, 'Age'] = 0\n",
        "\n",
        "print(train_df.head(40))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F5MwCSFe9GQo",
        "outputId": "155bc2a8-0698-41cf-ea0b-1e1441811f9e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "    PassengerId  Survived  Pclass  \\\n",
            "0             1         0       3   \n",
            "1             2         1       1   \n",
            "2             3         1       3   \n",
            "3             4         1       1   \n",
            "4             5         0       3   \n",
            "5             6         0       3   \n",
            "6             7         0       1   \n",
            "7             8         0       3   \n",
            "8             9         1       3   \n",
            "9            10         1       2   \n",
            "10           11         1       3   \n",
            "11           12         1       1   \n",
            "12           13         0       3   \n",
            "13           14         0       3   \n",
            "14           15         0       3   \n",
            "15           16         1       2   \n",
            "16           17         0       3   \n",
            "17           18         1       2   \n",
            "18           19         0       3   \n",
            "19           20         1       3   \n",
            "20           21         0       2   \n",
            "21           22         1       2   \n",
            "22           23         1       3   \n",
            "23           24         1       1   \n",
            "24           25         0       3   \n",
            "25           26         1       3   \n",
            "26           27         0       3   \n",
            "27           28         0       1   \n",
            "28           29         1       3   \n",
            "29           30         0       3   \n",
            "30           31         0       1   \n",
            "31           32         1       1   \n",
            "32           33         1       3   \n",
            "33           34         0       2   \n",
            "34           35         0       1   \n",
            "35           36         0       1   \n",
            "36           37         1       3   \n",
            "37           38         0       3   \n",
            "38           39         0       3   \n",
            "39           40         1       3   \n",
            "\n",
            "                                                 Name  Sex  Age  SibSp  Parch  \\\n",
            "0                             Braund, Mr. Owen Harris    1  2.0      1      0   \n",
            "1   Cumings, Mrs. John Bradley (Florence Briggs Th...    0  0.0      1      0   \n",
            "2                              Heikkinen, Miss. Laina    0  2.0      0      0   \n",
            "3        Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  2.0      1      0   \n",
            "4                            Allen, Mr. William Henry    1  2.0      0      0   \n",
            "5                                    Moran, Mr. James    1  NaN      0      0   \n",
            "6                             McCarthy, Mr. Timothy J    1  3.0      0      0   \n",
            "7                      Palsson, Master. Gosta Leonard    1  0.0      3      1   \n",
            "8   Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)    0  2.0      0      2   \n",
            "9                 Nasser, Mrs. Nicholas (Adele Achem)    0  1.0      1      0   \n",
            "10                    Sandstrom, Miss. Marguerite Rut    0  0.0      1      1   \n",
            "11                           Bonnell, Miss. Elizabeth    0  3.0      0      0   \n",
            "12                     Saundercock, Mr. William Henry    1  2.0      0      0   \n",
            "13                        Andersson, Mr. Anders Johan    1  2.0      1      5   \n",
            "14               Vestrom, Miss. Hulda Amanda Adolfina    0  1.0      0      0   \n",
            "15                   Hewlett, Mrs. (Mary D Kingcome)     0  3.0      0      0   \n",
            "16                               Rice, Master. Eugene    1  0.0      4      1   \n",
            "17                       Williams, Mr. Charles Eugene    1  NaN      0      0   \n",
            "18  Vander Planke, Mrs. Julius (Emelia Maria Vande...    0  2.0      1      0   \n",
            "19                            Masselmani, Mrs. Fatima    0  NaN      0      0   \n",
            "20                               Fynney, Mr. Joseph J    1  2.0      0      0   \n",
            "21                              Beesley, Mr. Lawrence    1  2.0      0      0   \n",
            "22                        McGowan, Miss. Anna \"Annie\"    0  1.0      0      0   \n",
            "23                       Sloper, Mr. William Thompson    1  2.0      0      0   \n",
            "24                      Palsson, Miss. Torborg Danira    0  0.0      3      1   \n",
            "25  Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...    0  2.0      1      5   \n",
            "26                            Emir, Mr. Farred Chehab    1  NaN      0      0   \n",
            "27                     Fortune, Mr. Charles Alexander    1  0.0      3      2   \n",
            "28                      O'Dwyer, Miss. Ellen \"Nellie\"    0  NaN      0      0   \n",
            "29                                Todoroff, Mr. Lalio    1  NaN      0      0   \n",
            "30                           Uruchurtu, Don. Manuel E    1  2.0      0      0   \n",
            "31     Spencer, Mrs. William Augustus (Marie Eugenie)    0  0.0      1      0   \n",
            "32                           Glynn, Miss. Mary Agatha    0  NaN      0      0   \n",
            "33                              Wheadon, Mr. Edward H    1  4.0      0      0   \n",
            "34                            Meyer, Mr. Edgar Joseph    1  0.0      1      0   \n",
            "35                     Holverson, Mr. Alexander Oskar    1  2.0      1      0   \n",
            "36                                   Mamee, Mr. Hanna    1  NaN      0      0   \n",
            "37                           Cann, Mr. Ernest Charles    1  2.0      0      0   \n",
            "38                 Vander Planke, Miss. Augusta Maria    0  1.0      2      0   \n",
            "39                        Nicola-Yarred, Miss. Jamila    0  1.0      1      0   \n",
            "\n",
            "              Ticket      Fare        Cabin Embarked  \n",
            "0          A/5 21171    2.0000          NaN      0.0  \n",
            "1           PC 17599   71.2833          C85      1.0  \n",
            "2   STON/O2. 3101282    2.0000          NaN      0.0  \n",
            "3             113803    1.0000         C123      0.0  \n",
            "4             373450    2.0000          NaN      0.0  \n",
            "5             330877    2.0000          NaN      2.0  \n",
            "6              17463    1.0000          E46      0.0  \n",
            "7             349909    2.0000          NaN      0.0  \n",
            "8             347742    2.0000          NaN      0.0  \n",
            "9             237736    2.0000          NaN      1.0  \n",
            "10           PP 9549    2.0000           G6      0.0  \n",
            "11            113783    2.0000         C103      0.0  \n",
            "12         A/5. 2151    2.0000          NaN      0.0  \n",
            "13            347082    2.0000          NaN      0.0  \n",
            "14            350406    2.0000          NaN      0.0  \n",
            "15            248706    2.0000          NaN      0.0  \n",
            "16            382652    2.0000          NaN      2.0  \n",
            "17            244373    2.0000          NaN      0.0  \n",
            "18            345763    2.0000          NaN      0.0  \n",
            "19              2649    2.0000          NaN      1.0  \n",
            "20            239865    2.0000          NaN      0.0  \n",
            "21            248698    2.0000          D56      0.0  \n",
            "22            330923    2.0000          NaN      2.0  \n",
            "23            113788    2.0000           A6      0.0  \n",
            "24            349909    2.0000          NaN      0.0  \n",
            "25            347077    2.0000          NaN      0.0  \n",
            "26              2631    2.0000          NaN      1.0  \n",
            "27             19950  263.0000  C23 C25 C27      0.0  \n",
            "28            330959    2.0000          NaN      2.0  \n",
            "29            349216    2.0000          NaN      0.0  \n",
            "30          PC 17601    2.0000          NaN      1.0  \n",
            "31          PC 17569  146.5208          B78      1.0  \n",
            "32            335677    2.0000          NaN      2.0  \n",
            "33        C.A. 24579    2.0000          NaN      0.0  \n",
            "34          PC 17604   82.1708          NaN      1.0  \n",
            "35            113789    1.0000          NaN      0.0  \n",
            "36              2677    2.0000          NaN      1.0  \n",
            "37        A./5. 2152    2.0000          NaN      0.0  \n",
            "38            345764    2.0000          NaN      0.0  \n",
            "39              2651    2.0000          NaN      1.0  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "중간중간에 Fare 부분이 제대로 바뀌지 않은 부분들이 있는데 이건 뭔지..?"
      ],
      "metadata": {
        "id": "j3zcDICU-8tZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#train_df['Age'].isnull().sum()\n",
        "#test_df['Age'].isnull().sum()\n",
        "\n",
        "#train_df['Age'].fillna(train_df.groupby)"
      ],
      "metadata": {
        "id": "h5Sb4mBzJnIr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#train_df['Age']= pd.qcut(train_df['Age'],5)\n",
        "#test_df['Age']= pd.qcut(test_df['Age'],5)\n",
        "\n",
        "#train_df['Age'].value_counts()"
      ],
      "metadata": {
        "id": "1YtojvRpHDBc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        ">### 산점도(Scattor Plot) \n",
        ">##### matplotlib.pyplot 모듈의 scatter() 함수를 이용하면 산점도를 그릴 수 있다"
      ],
      "metadata": {
        "id": "_v-aZf9mybJ1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#plt.scatter(df['Age'])\n",
        "#plt.show()"
      ],
      "metadata": {
        "id": "F7jLyJQDx7c0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def bar_chart(feature):\n",
        "  survived = train_df[train_df['Survived']==1][feature].value_counts()\n",
        "  dead = train_df[train_df['Survived']==0][feature].value_counts()\n",
        "  df = pd.DataFrame([survived, dead])\n",
        "  df.index = ['Surevived', 'Dead']\n",
        "  df.plot(kind = 'bar', stacked = True, figsize = (10,5))\n",
        "\n",
        "  # 입력값을 받아들일 때 여러 데이터에 적용할 수 있게 코드를 다시 짜라..\n",
        "  # train_df 외에 다른 데이터도 바꿀 수 있게 해라"
      ],
      "metadata": {
        "id": "8D1xIlkCDyfW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 이거저거 해보다가 다른 분 것을 봤는데 이것도 괜찮은거 같아서 짜본다..\n",
        "\n",
        "plt.figure(figsize = (10,5))\n",
        "plt.title('나이',fontsize = 20)\n",
        "sns.stripplot(x = 'Survived', y = 'Age', data = train_df, jitter = True)\n",
        "plt.xlabel('생존여부', fontsize = 20)\n",
        "plt.ylabel('나이', fontsize = 20)\n",
        "\n",
        "\n",
        "plt.show()\n",
        "\n",
        "#0: 사망, 1: 생존"
      ],
      "metadata": {
        "id": "ZzZwD-Cp0X3J",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 364
        },
        "outputId": "77e165c2-e0d1-48d9-8f18-4aacd19c3762"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Parch(함께 탑승한 부모, 자식의 수)\n",
        "bar_chart('Parch')"
      ],
      "metadata": {
        "id": "rrIU54FgxQLt",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 359
        },
        "outputId": "3c0703bb-ac6c-4719-b447-ebfe3931b872"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Parch 별 생존자 수치 분석\n",
        "train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived')"
      ],
      "metadata": {
        "id": "tN1Hi4Gib5qu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "outputId": "95263576-c2a2-4ba3-b655-ffdcaff9ce99"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Parch  Survived\n",
              "4      4  0.000000\n",
              "6      6  0.000000\n",
              "5      5  0.200000\n",
              "0      0  0.343658\n",
              "2      2  0.500000\n",
              "1      1  0.550847\n",
              "3      3  0.600000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-7868ba45-bf93-487d-b190-128c58182935\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Parch</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>6</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>5</td>\n",
              "      <td>0.200000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.343658</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>0.500000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.550847</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>0.600000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7868ba45-bf93-487d-b190-128c58182935')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-7868ba45-bf93-487d-b190-128c58182935 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-7868ba45-bf93-487d-b190-128c58182935');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 459
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Fare(티켓 요금)\n",
        "bar_chart('Fare')"
      ],
      "metadata": {
        "id": "ZYb1kpw2xZvW",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 876
        },
        "outputId": "fb8a321b-f9b0-4d65-b24f-604f67662f9d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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th4HEd093gokIiIishAGKyIiIiILYbAiIiIishAGKyIiIiILYbAiIiIishAGKyIiIiILYbAiIqsVGxsrUygUGqVSqYmKivKvrq4WMjMzO2i1WpWvr2/wsGHDutXW1gottf3HP/7R1dfXN1gulwdv377dsbXHTkTtEzcIJaK79sCnD4Rasr8zk88k3apObm6uzerVq92zsrLSHBwczJGRkd3Wrl3rsn//fqfZs2cXP/vss2UxMTG+y5cvd124cKHh2rZJSUl2X331lUtWVtZZvV5vM3jw4IAnnngiTSLhn0QiujtcsSIiq9XY2ChUVVWJjEYjampqRF5eXsYTJ05Ip06dWgYA06ZNK929e7fzje3i4+Odo6OjL3Xs2NGsUqnq/fz86o4cOcIzAYnorjFYEZFV8vf3N86aNavI399fK5PJdFKptLFv377VUqm00cbm6nF/crm8vri4uMONbQsLCzv4+PjUN7339PSsz8/Pv6keEdGdYrAiIqtkMBjEe/fudc7JyTlTVFSUWl1dLdqxYweflSKiNsUHCojIKu3evdvR19e3ztPTswEARo4cWX7s2DGHyspKsdFohI2NDfLy8jq4u7vX39jWy8vruhWqCxcuXLeCRUT0R3HFioisklwur09OTnaorKwUmUwmHDp0SKpWq2v79OlTuWHDhs4AsH79+i7Dhw8vv7Htk08+Wf7VV1+51NTUCJmZmR3y8vLsHn300arWnwURtTcMVkRklSIiIqqioqLKtFptUGBgoMZkMgnz5883vPfeewUffvhhV19f3+CysjLJnDlzSgBgy5YtTnPnzvUEgLCwsNqRI0deCggI0AwdOjTg3//+t56/CCQiSxDMZnNbj4GIrExKSkqeTqcraetxtBcpKSmuOp1O3tbjIKK7xxUrIiIiIgthsCIiIiKyEAYrIiIiIgthsCIiIiKyEAYrIiIiIgthsCIiIiKyEAYrIrJasbGxMoVCoVEqlZqoqCj/6upq4Y033nDz9fUNFgQh9OLFi7+7OdXMmTO9FQqFplu3bpopU6b4mEwmAECvXr0C5XJ5sEqlUqtUKnVhYaEEALKzszv07t07ICgoSB0QEKCOi4tzAoDDhw/bN9UNDAxUb9q06aZDn4no/sEd8YjormWogkIt2V9QZkbSrerk5ubarF692j0rKyvNwcHBHBkZ2W3t2rUu/fv3v/Lk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mWuvXr0qXVxcTCaTCePHj/cLCAioXbx4cbGl50hE1oXBiois1ogRI7p3795dM3z4cMX777//i6ura+OmTZuc3d3dtadPn+40atQo5cMPP6wEgLy8PJv+/fsrAOC3X/L9MnTo0AClUqkZOXLkpbCwsOZf8cXHx7vExMRcF9JOnz5tp1KpNHK5PPjbb791Wr16dT4AHDhwwGHnzp1dEhISpE3bNMTFxTm15vdARPcOwWw237oWEdE1UlJS8nQ6XUlbj6O9SElJcdXpdPK2HgcR3T2uWBERERFZCIMVERERkYUwWBERERFZCIMVERERkYUwWBERERFZCIMVERERkYUwWBGR1XrttddkSqVSo1AoNEuWLJEBwJw5czwDAgLUKpVK3a9fP2VeXp5NS23FYnFo075TERERitYdORG1V9zHioju2E37WC12CrXoByy+nHSrKidPnrSLiYnpnpycnGFnZ2fq379/wOrVq/Wenp7GprP6/vWvf8nS09PtPv/8819ubG9vb9+jurr6vxYd9x/EfayI2g+uWBGRVTpz5kzHHj16XJFKpSYbGxv069evctu2bc7XHoBcVVUlEgShLYdJRPcZBisiskohISE1iYmJ0qKiInFlZaXowIEDTvn5+R0A4Pnnn/fq2rWrNj4+vss777xzoaX29fX1ouDg4CCdTqfavHmzc+uOnojaKwYrIrJKPXv2rJ0zZ07RwIEDAwYMGKDUaDTVYrEYAPDhhx8WFhUVpY4ePbr0nXfekbXUPjs7OzUtLS1j69at5xctWuRz9uxZ21adABG1SwxWRGS15s2bV3L27NmMU6dOZXXu3LkxICCg9trr06ZNu7Rnz57OLbX19/c3AoBara7v06dPZWJion1rjJmI2jcGKyKyWoWFhRIAyM7O7rB3717n6dOnXzpz5kzzytMXX3zh3L1795ob2xkMBnFNTY0AABcvXpScOnXKQavV3lSPiOhOSdp6AEREf9SIESO6l5eXSyQSifn999//xdXVtXHixIny8+fP2wmCYPb29q5ft26dHgCOHj1q//HHH7vFxcXpT58+bTdr1iw/QRBgNpsxd+7cotDQ0NpbfR4R0a1wuwUiumM3bbdAd4XbLRC1H7wVSERERGQhDFZEREREFsJgRURERGQhDFZEREREFsJgRURERGQhDFZEREREFsJgRURWacyYMXIXFxed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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize = (10,5))\n",
        "plt.title('티켓 요금',fontsize = 20)\n",
        "sns.stripplot(x = 'Survived', y = 'Fare', data = train_df, jitter = True)\n",
        "plt.xlabel('생존여부', fontsize = 20)\n",
        "plt.ylabel('요금', fontsize = 20)\n",
        "\n",
        "\n",
        "plt.show()\n",
        "\n",
        "#0: 사망, 1: 생존"
      ],
      "metadata": {
        "id": "LPloZ_OeEAVE",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 364
        },
        "outputId": "5ade68a9-72b9-4955-af3f-282b42017264"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">##### 싸게 티켓을 주고 살아남은 사람들이 많지만, 비싼 티켓보다 싼 값의 티켓의 양이 더 많으므로 유의미한 결과를 내진 못했다고 생각한다"
      ],
      "metadata": {
        "id": "N_65k2K-IM8t"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. 데이터 전처리\n",
        "#### https://whitewing4139.tistory.com/178 의 이론을 참고"
      ],
      "metadata": {
        "id": "bwEQgQkuHM-H"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "1) 필요 없는 데이터 제거\n",
        "\n",
        "##### Ticket과 Cabin의 데이터는 필요없을거라 판단하여 해당 열을 제거한다"
      ],
      "metadata": {
        "id": "-tTSJY3rdPtZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "combine = [train_df, test_df]\n",
        "\n",
        "print(\"Before\", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)\n",
        "\n",
        "train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)\n",
        "test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)\n",
        "combine = [train_df, test_df]\n",
        "\n",
        "print(\"After\", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)"
      ],
      "metadata": {
        "id": "FPjWFlbGdYCI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "63734bc2-fee2-43a2-dcf6-5e6bf4f07a22"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Before (891, 12) (418, 11) (891, 12) (418, 11)\n",
            "After (891, 10) (418, 9) (891, 10) (418, 9)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "쓰지 않을 변수 Name과 Passengerld를 제거한다\n"
      ],
      "metadata": {
        "id": "TBWIirxYz5Bt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df = train_df.drop(['Name', 'PassengerId'], axis=1)\n",
        "test_df = test_df.drop(['Name'], axis=1)\n",
        "combine = [train_df, test_df]\n",
        "train_df.shape, test_df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NmD-OZQ20Bjk",
        "outputId": "6ea5d51d-f14a-4886-cf17-0fe0d373af4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((891, 8), (418, 8))"
            ]
          },
          "metadata": {},
          "execution_count": 463
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "VIkrDVbr0I8G",
        "outputId": "505412a6-c9ce-4e03-a00d-ee68e6dfe6b0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    1  2.0      1      0   2.0000      0.0\n",
              "1         1       1    0  0.0      1      0  71.2833      1.0\n",
              "2         1       3    0  2.0      0      0   2.0000      0.0\n",
              "3         1       1    0  2.0      1      0   1.0000      0.0\n",
              "4         0       3    1  2.0      0      0   2.0000      0.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-895ffae6-a755-428d-b8a2-1d22847e6cc9\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-895ffae6-a755-428d-b8a2-1d22847e6cc9')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-895ffae6-a755-428d-b8a2-1d22847e6cc9 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-895ffae6-a755-428d-b8a2-1d22847e6cc9');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 464
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "sex가 NaN으로 처리되어 있는 것을 확인할 수 있었다 -> 오류 고침"
      ],
      "metadata": {
        "id": "lrpSQZyS0Wz2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lX8uFyew0aWs",
        "outputId": "b9bd5c22-466c-4c65-e886-f0a2246ad7f3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Survived      0\n",
              "Pclass        0\n",
              "Sex           0\n",
              "Age         162\n",
              "SibSp         0\n",
              "Parch         0\n",
              "Fare          0\n",
              "Embarked      0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 465
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "VLJ6Z71H5bSN",
        "outputId": "8f5a77c6-91ec-427b-fbe8-78e70981b058"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    1  2.0      1      0   2.0000      0.0\n",
              "1         1       1    0  0.0      1      0  71.2833      1.0\n",
              "2         1       3    0  2.0      0      0   2.0000      0.0\n",
              "3         1       1    0  2.0      1      0   1.0000      0.0\n",
              "4         0       3    1  2.0      0      0   2.0000      0.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a1bd0f92-088e-475e-9381-e33594d91793\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a1bd0f92-088e-475e-9381-e33594d91793')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-a1bd0f92-088e-475e-9381-e33594d91793 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a1bd0f92-088e-475e-9381-e33594d91793');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 466
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 1) 이상치 및 결측치 처리\n",
        ">##### * pandas.DataFrame.isnull() : 결측치 여부를 T/F값으로 반환해준다. 결측치인 경우 True를 반환한다.\n",
        ">##### * pandas.DataFrame.notnull() : 결측치 여부를 T/F값으로 반환한다. 실측치인 경우엔 True로 반환한다.\n",
        ">##### * pandas.DataFrame.dropna(axis): 결측치가 포함된 데이터를 제외하고 추출한다.\n",
        ">##### * pandas.DataFrame.fillna(\"대체값\") : 결측치 데이터를 \"대체값\"으로 적용한다.\n",
        ">>##### - fillna() 메서드의 method인자값을 [\"ffill\", \"pad\"] 중 하나로 지정하여, 결측치 앞의 값으로 대체할 수 있다.\n",
        ">>##### - fillna() 메서드의 method인자값을 [\"bfill\", \"pad\"] 중 하나로 지정하여\n",
        ">##### * numpy.where(\"조건\", \"조건 True시 대체값\", \"조건 False 시 대체값\")\n",
        "\n",
        ">##### * 이상치(Outlier): 수집된 데이터의 모음(데이터 셋) 값 중, 다른 데이터에 비해 매우 크거나 작은 값\n",
        ">>##### 1. 이상치 데이터의 처리(4)\n",
        ">>##### - 이상치의 삭제\n",
        ">>##### - 이상치의 값 대체\n",
        ">>##### - 데이터 셋의 축소/과장\n",
        ">>##### - 데이터 셋의 최소최대척도 적용\n",
        ">>##### - 데이터 셋의 정규화\n",
        "\n",
        ">##### * 결측치(Missing Value): 데이터 수집 과정에서 측정되지 않거나 누락된 데이터\n"
      ],
      "metadata": {
        "id": "MqKPwurlLAFJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df['Age'] = train_df['Age'].fillna(29)  \n",
        "# 결측치를 채우기로 결정, 결측치를 대신하여 넣고자 하는 값을 명시 - 최빈값을 넣기로 하였다\n",
        "\n",
        "test_df['Age'] = test_df['Age'].fillna(29)\n",
        "test_df['Fare'] = test_df['Fare'].fillna(32)\n",
        "\n",
        "# 나이는 어린애 노인들이 많은데, 특정한 쪽으로 쏠릴 경우가 많다 따라서 나이는 평균값\n",
        "# 서로 연관이 없는 데이터는 최빈값 -> 데이터 형식에 따라 어떤 식으로 전처리 할지 알아보기"
      ],
      "metadata": {
        "id": "bCDzl_XhEcjc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df_x = train_df[['Pclass', 'SibSp','Age', 'Sex', 'Fare']]\n",
        "train_df_y = train_df['Survived']"
      ],
      "metadata": {
        "id": "3DGpH6oAyznl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_df_x = test_df[['Pclass', 'SibSp', 'Age', 'Sex', 'Fare']]"
      ],
      "metadata": {
        "id": "k-U1-0Ep5ti0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p0yPsfRn4Lxo",
        "outputId": "c465c842-758b-41e2-d9a3-604dbed416f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Survived    0\n",
              "Pclass      0\n",
              "Sex         0\n",
              "Age         0\n",
              "SibSp       0\n",
              "Parch       0\n",
              "Fare        0\n",
              "Embarked    0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 470
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "_ELa8Miv4gL-",
        "outputId": "95d1a0f2-15f0-4626-cc6d-69f2534b5735"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    1  2.0      1      0   2.0000      0.0\n",
              "1         1       1    0  0.0      1      0  71.2833      1.0\n",
              "2         1       3    0  2.0      0      0   2.0000      0.0\n",
              "3         1       1    0  2.0      1      0   1.0000      0.0\n",
              "4         0       3    1  2.0      0      0   2.0000      0.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8efcbbee-99fd-4e06-869e-52a7e1265c85\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2.0000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8efcbbee-99fd-4e06-869e-52a7e1265c85')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-8efcbbee-99fd-4e06-869e-52a7e1265c85 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-8efcbbee-99fd-4e06-869e-52a7e1265c85');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 471
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 2) 표준화 및 정규화\n",
        ">##### * 정규화: 특성 값의 범위를 [0,1]로 옮긴다\n",
        ">##### - 식: (X-MIN)/(MAX-MIN)\n",
        ">>##### # 수식을 통한 정규화\n",
        ">>##### # sklearn.preprocessing을 통한 정규화\n",
        ">##### * 표준화: 어떤 특성의 값들이 정규분포를 따른다고 가정하고, 값들을 0의 평균, 1의 표준편차를 갖도록 반환한다\n",
        ">##### - 식: (X-평균)/표준편차\n",
        ">>##### # numpy를 이용한 표준화\n",
        ">>##### # zscore를 이용한 표준화\n",
        ">>##### # sckit-learn processing을 이용한 표준화"
      ],
      "metadata": {
        "id": "RHtQogDT-HG8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "##표준화 : 평균 = 0, 표준편차 = 1\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "\n",
        "# StandardScaler객체 생성\n",
        "scaler = StandardScaler() \n",
        "scaler.fit(train_df_x)\n",
        "train_scaled = scaler.transform(train_df_x)\n",
        "train_x = train_scaled\n",
        "\n",
        "scaler.fit(test_df_x)\n",
        "test_scaled = scaler.transform(test_df_x)\n",
        "test_df_x = test_scaled\n",
        "\n",
        "print(test_df_x)"
      ],
      "metadata": {
        "id": "Z_zod3K57GfU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dab8c2b3-df4d-463b-8c85-9f5c23132369"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 0.87348191 -0.49947002 -0.44833988  0.75592895 -0.34753895]\n",
            " [ 0.87348191  0.61699237 -0.44833988 -1.32287566 -0.34753895]\n",
            " [-0.31581919 -0.49947002 -0.35754236  0.75592895 -0.34753895]\n",
            " ...\n",
            " [ 0.87348191 -0.49947002 -0.44833988  0.75592895 -0.34753895]\n",
            " [ 0.87348191 -0.49947002  2.00319299  0.75592895 -0.34753895]\n",
            " [ 0.87348191  0.61699237  2.00319299  0.75592895 -0.34753895]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "##정규화 : [0,1]\n",
        "\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "\n",
        "# MinMaxScaler객체 생성\n",
        "scaler = MinMaxScaler()\n",
        "  \n",
        "scaler.fit(train_df_x)\n",
        "train_scaled = scaler.transform(train_df_x)\n",
        "\n",
        "train_df_x = train_scaled \n",
        "\n",
        "scaler.fit(test_df_x)\n",
        "test_scaled = scaler.transform(test_df_x)\n",
        "\n",
        "test_df_x = test_scaled \n",
        "\n",
        "\n",
        "print(test_df_x)"
      ],
      "metadata": {
        "id": "-WT9hWob_tuT",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a65fddb7-fb6e-4643-a6f8-04876e9b1f89"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1.         0.         0.06896552 1.         0.00195569]\n",
            " [1.         0.125      0.06896552 0.         0.00195569]\n",
            " [0.5        0.         0.10344828 1.         0.00195569]\n",
            " ...\n",
            " [1.         0.         0.06896552 1.         0.00195569]\n",
            " [1.         0.         1.         1.         0.00195569]\n",
            " [1.         0.125      1.         1.         0.00195569]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. 모델 정의 및 학습\n",
        "\n",
        "### 모델은 몇가지 아는게 없고, 어떤 것을 써야할 지 몰라서 다른 분들은 어떤걸 하셨나 찾아보니 의사결정나무, 로지스틱회귀, 랜덤포레스트,LightGBM, 인공신경망 모델들을 사용할 수 있는 것을 알 수 있었습니다."
      ],
      "metadata": {
        "id": "ey_5qu7IIS3h"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델 평균 정확도 구하기\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.model_selection import cross_val_score\n",
        "k_fold = KFold(n_splits=10, shuffle=True, random_state=0)"
      ],
      "metadata": {
        "id": "mveVZnQlH3f0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# kfold 함수\n",
        "\n",
        "def kfold_func(clf):\n",
        "\n",
        "  scoring = 'accuracy'\n",
        "  score = cross_val_score(clf, train_df_x, train_df_y, cv=k_fold, n_jobs=1, scoring=scoring)\n",
        "  print(score)\n",
        "  #print('평균 정확도:', round(np.mean(score)*100,2))\n",
        "  \n",
        "# 프린트 케이 폴드 펑션.. 써서 값넣기"
      ],
      "metadata": {
        "id": "yRFoqZrjuZWB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### kfold함수에서 설정할 수 있는 argument 목록\n",
        ">##### * n_splits: 분할한 세트의 개수, 1세트의 test만 데이터로 활용하고 나머지 데이터는 train 데이터로 사용, 5가 기본값으로 주어져 있다.\n",
        ">##### * shuffle: True로 설정 시 데이터셋 내의 순서를 섞어 셈플링, False 인 경우 순서대로 분할, False가 기본으로 주어져 있다\n",
        ">##### * random_state: seed 설정, 특정 정수로 지정시 샘플링 결과 고정"
      ],
      "metadata": {
        "id": "ry4Jd6qzvWTE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#from sklearn.model_selection import KFold\n",
        "\n",
        "#X = np.array(df.iloc[:, :-1]) # class 열 제외한 feature 열들 모음 -> array 변환\n",
        "#y = df['class']\n",
        "\n",
        "# split 개수, 셔플 여부 및 seed 설정\n",
        "#kf = KFold(n_splits = 10, shuffle = False, random_state = 0)\n",
        "\n",
        "# split 개수 스텝 만큼 train, test 데이터셋을 매번 분할\n",
        "#for train_df_index, test_df_index in kf.split(X):\n",
        "#    train_df_x, test_df_x = X[train_df_index], X[test_df_index]\n",
        "#    train_df_y, test_df_y = y[train_df_index], y[test_df_index]"
      ],
      "metadata": {
        "id": "spbW8iBWw72F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#from sklearn.ensemble import RandomForestClassifier\n",
        "#from sklearn.metrics import accuracy_score\n",
        "\n",
        "#accuracy_history = []\n",
        "\n",
        "#for train_df_index, test_df_index in kf.split(X):\n",
        "#   train_df_x, test_df_x = X[train_df_index], X[test_df_index]\n",
        "#    train_df_y, test_df_y = y[train_df_index], y[test_df_index]\n",
        "\n",
        "#    model = RandomForestClassifier(n_estimators=5, random_state=0) # 모델 선언\n",
        "#    model.fit(train_df_x, train_df_y) # 모델 학습\n",
        "\n",
        "#    y_pred = model.predict(test_df_x) # 예측 라벨\n",
        "#    accuracy_history.append(accuracy_score(y_pred, test_df_y)) # 정확도 측정 및 기록\n",
        "\n",
        "#print(\"각 분할의 정확도 기록 :\", accuracy_history)\n",
        "#print(\"평균 정확도 :\", np.mean(accuracy_history))"
      ],
      "metadata": {
        "id": "8JX1p8asulIX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = train_df.drop(\"Survived\", axis=1)\n",
        "Y_train = train_df[\"Survived\"]\n",
        "X_test  = test_df.drop(\"PassengerId\", axis=1).copy()\n",
        "X_train.shape, Y_train.shape, X_test.shape"
      ],
      "metadata": {
        "id": "_3jL_AmMy80I",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "862256cb-7012-482d-fc47-023ae3fc9f4d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((891, 7), (891,), (418, 7))"
            ]
          },
          "metadata": {},
          "execution_count": 478
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.linear_model import SGDClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier"
      ],
      "metadata": {
        "id": "9G9XKJkc_Pat"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Logistic Regression\n",
        "\n",
        "logreg = LogisticRegression()\n",
        "logreg.fit(X_train, Y_train)\n",
        "Y_pred = logreg.predict(X_test)\n",
        "acc_log = round(logreg.score(X_train, Y_train) * 100, 2)\n",
        "acc_log"
      ],
      "metadata": {
        "id": "q5qDHEQVJiPU",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 400
        },
        "outputId": "4095c7d1-85e3-4d86-c2e4-61cff4f6faca"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-480-5854ca91fc64>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mlogreg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLogisticRegression\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mY_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mY_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0macc_log\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscore\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mY_train\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m   1506\u001b[0m             \u001b[0m_dtype\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1507\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1508\u001b[0;31m         X, y = self._validate_data(\n\u001b[0m\u001b[1;32m   1509\u001b[0m             \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1510\u001b[0m             \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    579\u001b[0m                 \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    580\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 581\u001b[0;31m                 \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    582\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    583\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[1;32m    962\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"y cannot be None\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    963\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 964\u001b[0;31m     X = check_array(\n\u001b[0m\u001b[1;32m    965\u001b[0m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    966\u001b[0m         \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0maccept_sparse\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    744\u001b[0m                     \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcasting\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"unsafe\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    745\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 746\u001b[0;31m                     \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    747\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mcomplex_warning\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    748\u001b[0m                 raise ValueError(\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m__array__\u001b[0;34m(self, dtype)\u001b[0m\n\u001b[1;32m   1991\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1992\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__array__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mNpDtype\u001b[0m \u001b[0;34m|\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndarray\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1993\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1994\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1995\u001b[0m     def __array_wrap__(\n",
            "\u001b[0;31mValueError\u001b[0m: could not convert string to float: 'S'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Support Vector Machines\n",
        "\n",
        "svc = SVC()\n",
        "svc.fit(X_train, Y_train)\n",
        "Y_pred = svc.predict(X_test)\n",
        "acc_svc = round(svc.score(X_train, Y_train) * 100, 2)\n",
        "acc_svc"
      ],
      "metadata": {
        "id": "QTtlXAXSJkbg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# KNN\n",
        "\n",
        "knn = KNeighborsClassifier(n_neighbors = 3)\n",
        "knn.fit(X_train, Y_train)\n",
        "Y_pred = knn.predict(X_test)\n",
        "acc_knn = round(knn.score(X_train, Y_train) * 100, 2)\n",
        "acc_knn"
      ],
      "metadata": {
        "id": "TpAb4_Ur_hMh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Stochastic Gradient Descent\n",
        "\n",
        "sgd = SGDClassifier()\n",
        "sgd.fit(X_train, Y_train)\n",
        "Y_pred = sgd.predict(X_test)\n",
        "acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)\n",
        "acc_sgd"
      ],
      "metadata": {
        "id": "DD3AgNVA_mBr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Decision Tree\n",
        "\n",
        "decision_tree = DecisionTreeClassifier()\n",
        "decision_tree.fit(X_train, Y_train)\n",
        "Y_pred = decision_tree.predict(X_test)\n",
        "acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)\n",
        "acc_decision_tree"
      ],
      "metadata": {
        "id": "1-T97P9N_oOB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Random Forest\n",
        "\n",
        "random_forest = RandomForestClassifier(n_estimators=100)\n",
        "random_forest.fit(X_train, Y_train)\n",
        "Y_pred = random_forest.predict(X_test)\n",
        "random_forest.score(X_train, Y_train)\n",
        "acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)\n",
        "acc_random_forest"
      ],
      "metadata": {
        "id": "KOxOZH6e_qRK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "models = pd.DataFrame({\n",
        "    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', \n",
        "              'Random Forest','Stochastic Gradient Decent',  'Decision Tree'],\n",
        "    'Score': [acc_svc, acc_knn, acc_log, \n",
        "              acc_random_forest, acc_sgd, acc_decision_tree]})\n",
        "models.sort_values(by='Score', ascending=False)"
      ],
      "metadata": {
        "id": "Ccep5rXt_wCE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "submission = pd.read_csv('/content/drive/MyDrive/All-in/titanic/gender_submission.csv')"
      ],
      "metadata": {
        "id": "iO3vdUjS_yPa"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
