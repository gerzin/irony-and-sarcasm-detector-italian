#!/usr/bin/env bash

#
# Downloads and extracts the tensorflow weights for the BERT_XXL_CASED model.
#

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PRETRAINED_MODELS_PATH="$SCRIPTPATH/../models/pretrained"

BERT_XXL_CASED=https://schweter.eu/cloud/berts/bert-base-italian-xxl-cased.tar.gz
#BERT_XXL_UNCASED=wget https://schweter.eu/cloud/berts/bert-base-italian-xxl-uncased.tar.gz

WEIGHTS_DIR_PATH="$SCRIPTPATH/../weights"

if [ ! -d "$WEIGHTS_DIR_PATH" ]; then
    echo "$WEIGHTS_DIR_PATH does not exits. Creating it ..."
    mkdir "$WEIGHTS_DIR_PATH"
fi



echo "Downloading $BERT_XXL_CASED into $WEIGHTS_DIR_PATH ..."

curl --request GET -sL \
     --url $BERT_XXL_CASED\
     --output "$WEIGHTS_DIR_PATH/bert-base-italian-xxl-cased.tar.gz"

echo "Extracting the archive into $PRETRAINED_MODELS_PATH/bert-base-italian-xxl-cased"
if [ ! -d "$PRETRAINED_MODELS_PATH/bert-base-italian-xxl-cased" ]; then
  mkdir "$PRETRAINED_MODELS_PATH/bert-base-italian-xxl-cased"
fi

tar -xf "$WEIGHTS_DIR_PATH/bert-base-italian-xxl-cased.tar.gz" -C "$PRETRAINED_MODELS_PATH/bert-base-italian-xxl-cased"

echo "Bert weights extracted. Deleting weights folder and archive."
rm -rf "$WEIGHTS_DIR_PATH"
echo "Done."

