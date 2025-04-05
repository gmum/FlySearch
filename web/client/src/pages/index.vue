<template>
  <v-app>
    <v-app-bar
      elevation="1"
    >
      <v-toolbar-title>FlySearch demo</v-toolbar-title>
      <v-spacer />
      <v-btn
        variant="outlined"
        color="error"
        class="mr-5"
        @click="resetEnv"
      >
        Start a new game
      </v-btn>
    </v-app-bar>
    <v-main>
      <v-container>
        <v-card
          v-if="!started"
          variant="outlined"
        >
          <v-card-title>Your task</v-card-title>
          <v-card-subtitle>A similar prompt is given to the AI agent</v-card-subtitle>
          <v-card-text>
            <v-list>
              <v-list-item>You are in command of a UAV, tasked with finding a target.</v-list-item>
              <v-list-item>
                You should fly BELOW 10 meters above the object and then reply with "FOUND". Being lower
                (closer to
                the
                object) (like 9, 8, or less meters) is good, being higher than that (like 11, 12, or more meters) is
                bad.
              </v-list-item>
              <v-list-item>
                You may not be able to see the object in the first image, so you need to perform a careful
                search.
                Your
                performance will be evaluated based on whether the object was at most 10 meters below the drone when you
                replied with "FOUND". The object MUST be in your field of view when you reply with "FOUND". You must be
                centered on the object.
              </v-list-item>

              <v-list-item>
                There is a grid overlaid on each image you are presented with. It is meant to (roughly)
                communicate
                which
                point will be in drone's center of vision if you move in that direction. Note that height of the drone
                is
                not represented in the grid.
              </v-list-item>


              <v-list-item>
                To move the drone in a certain direction, use the following format: (x, y, z). For example,
                if you want
                to
                fly to the place denoted as (10, 10) on the grid without changing the altitude, you should reply with
                (10,
                10, 0).
              </v-list-item>

              <v-list-item>
                x and y are the coordinates on the grid, and z is the altitude difference. For example, (0,
                0, -10)
                means
                that you are moving 10 meters down. This is especially important, since you need to get close to the
                object in question. tag should contain the move you are making.
              </v-list-item>

              <v-list-item>
                If you find the object, fly below 10 meters relative to it and reply with "FOUND". Remember,
                it must be in
                your field of view when you reply with "FOUND" and you must be 10 meters above it or closer. Being too
                far
                away is not acceptable.
              </v-list-item>

              <v-list-item>
                You shouldn't move into coordinates that are outside of your view. Otherwise, you may hit
                something which
                is
                not ideal.
                You can make at most 10 moves. Your altitude cannot exceed 300 meters. Your search area is
                100x100m from the drone's starting position.
              </v-list-item>
            </v-list>

            <v-alert
              variant="text"
              color="primary"
            >
              To start a new game press the button in the top right corner.
            </v-alert>
          </v-card-text>
        </v-card>
        <v-card
          v-if="started"
          variant="outlined"
        >
          <v-card-title>Target: {{ target }}</v-card-title>
          <v-card-subtitle>
            Move the UAV to locate the target, then click FOUND when the target is visible and altitude
            is less than 10m above target
          </v-card-subtitle>
          <v-card-text>
            <div class="d-flex flex-column align-center mt-1 mb-6">
              <v-img
                :width="500"
                :height="500"
                :src="'data:image/jpeg;base64,' + image_b64"
                class="mb-2"
              >
                <template #placeholder>
                  <div class="d-flex flex-column align-center justify-center fill-height">
                    <div>
                      <v-progress-circular
                        color="grey"
                        indeterminate
                        size="100"
                      />
                    </div>
                  </div>
                </template>
              </v-img>
              <div>Current altitude: {{ altitude }}</div>
              <div :class="collision === true ? 'text-red' : ''">
                Collided on the last action: {{ collision }}
              </div>
            </div>

            <v-form>
              <v-row>
                <v-col
                  cols="12"
                  sm="12"
                  md="5"
                  class="text-center"
                >
                  <div>If the object is visible and the altitude is less than 10m above the target</div>
                  <v-btn
                    variant="outlined"
                    :disabled="!image_b64"
                    class="mt-3"
                    color="success"
                    @click="action(true)"
                  >
                    Found
                  </v-btn>
                </v-col>
                <v-col
                  cols="12"
                  sm="12"
                  md="2"
                  class="text-center"
                >
                  <v-divider
                    vertical
                    class=""
                  >
                    Or
                  </v-divider>
                </v-col>
                <v-col
                  cols="12"
                  sm="12"
                  md="5"
                  class="text-center"
                >
                  <div class="text-center">
                    Else, move the UAV closer to the target
                  </div>
                  <div class="d-flex flex-row justify-center align-center mt-2">
                    <v-number-input
                      v-model="x"
                      :disabled="!image_b64"
                      variant="outlined"
                      hide-spin-buttons
                      type="number"
                      label="X"
                      control-variant="stacked"
                      class="mr-2"
                      prcision="0"
                    />
                    <v-number-input
                      v-model="y"
                      :disabled="!image_b64"
                      variant="outlined"
                      hide-spin-buttons
                      type="number"
                      label="Y"
                      control-variant="stacked"
                      class="mr-2"
                      prcision="0"
                    />
                    <v-number-input
                      v-model="z"
                      :disabled="!image_b64"
                      variant="outlined"
                      hide-spin-buttons
                      type="number"
                      label="Z"
                      control-variant="stacked"
                      class="mr-2"
                      prcision="0"
                    />
                  </div>
                  <v-btn
                    :disabled="!image_b64"
                    variant="outlined"
                    color="primary"
                    class="mt-n2"
                    @click="action(false)"
                  >
                    Move
                  </v-btn>
                </v-col>
              </v-row>
            </v-form>
          </v-card-text>
        </v-card>
      </v-container>
    </v-main>
    <v-overlay
      v-model="error"
      class="align-center justify-center"
    >
      <v-alert
        :value="error"
        type="error"
        title="Error occurred!"
        :text="error_message"
        min-width="500"
      />
    </v-overlay>
  </v-app>
</template>

<script setup>
import {inject, ref} from "vue";

const axios = inject('axios');

const api_base = 'http://localhost:8000';

const x = ref(0);
const y = ref(0);
const z = ref(0);
const image_b64 = ref(null);
const collision = ref(null);
const altitude = ref(null);
const target = ref(null);
const started = ref(false);

const error = ref(false);
const error_message = ref('');

function cleanStatus() {
  image_b64.value = null;
  collision.value = null;
  altitude.value = null;
}

function getStatus() {
  axios.get(api_base + '/get_observation')
      .then((response) => {
        const current_status = response.data;
        image_b64.value = current_status.image_b64;
        collision.value = current_status.collision;
        altitude.value = current_status.altitude;
        x.value = 0
        y.value = 0
        z.value = 0
      })
      .catch((err) => {
        error.value = true;
        error_message.value = err.message;
        console.error(err);
      });
}

function resetEnv() {
  started.value = true;
  cleanStatus();
  target.value = null;
  axios.post(api_base + '/generate_new')
      .then((response) => {
        getStatus();
        target.value = response.data.target;
      })
      .catch((err) => {
        error.value = true;
        error_message.value = err.message;
        console.error(err);
      });
}

function action(is_done) {
  cleanStatus();

  const request = {
    found: is_done,
    coordinate_change: is_done ? [0, 0, 0] : [x.value, y.value, z.value]
  };

  axios.post(api_base + '/move', request)
      .then(() => {
        if (is_done) {
          resetEnv();
        } else {
          getStatus();
        }
      })
      .catch((err) => {
        error.value = true;
        error_message.value = err.message;
        console.error(err);
      });
}
</script>

<!--suppress CssUnusedSymbol -->
<style scoped>
.v-container {
  max-width: 1000px;
  margin-top: 1rem
}
</style>
