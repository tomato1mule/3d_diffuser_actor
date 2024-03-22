// Modified from https://unpkg.com/browse/three@0.160.1/examples/jsm/controls/FirstPersonControls.js

import {
	MathUtils,
	Spherical,
	Vector3
} from 'three';

const _lookDirection = new Vector3();
const _spherical = new Spherical();
const _target = new Vector3();

class CameraControls {

	constructor( object, domElement ) {

		object.up.set(0,0,1);

		this.object = object;
		this.domElement = domElement;

		// API

		this.enabled = true;

		this.movementSpeed = 1.0;
		this.lookSpeed = 0.005;

		this.lookVertical = true;
		this.autoForward = false;

		this.activeLook = true;

		this.constrainVertical = false;
		this.verticalMin = 0;
		this.verticalMax = Math.PI;

		this.mouseDragOn = false;

		// internals

		this.pointerX = 0;
		this.pointerY = 0;

		this.moveForward = false;
		this.moveBackward = false;
		this.moveLeft = false;
		this.moveRight = false;

		this.viewHalfX = 0;
		this.viewHalfY = 0;

		// private variables

		let lat = 0;
		let lon = 0;

		//

		this.handleResize = function () {

			if ( this.domElement === document ) {

				this.viewHalfX = window.innerWidth / 2;
				this.viewHalfY = window.innerHeight / 2;

			} else {

				this.viewHalfX = this.domElement.offsetWidth / 2;
				this.viewHalfY = this.domElement.offsetHeight / 2;

			}

		};

		this.onPointerDown = function ( event ) {

			if ( this.domElement !== document ) {

				this.domElement.focus();

			}

			if ( this.activeLook ) {

				switch ( event.button ) {

					case 0: this.moveForward = true; break;
					case 2: this.moveBackward = true; break;

				}

			}

			this.mouseDragOn = true;

		};

		this.onPointerUp = function ( event ) {

			if ( this.activeLook ) {

				switch ( event.button ) {

					case 0: this.moveForward = false; break;
					case 2: this.moveBackward = false; break;

				}

			}

			this.mouseDragOn = false;

		};

		this.onPointerMove = function ( event ) {

			if ( this.domElement === document ) {

				let pointerX = event.pageX - this.viewHalfX;
				let pointerY = event.pageY - this.viewHalfY;
				if (Math.abs(pointerX) < 0.5*(this.viewHalfX)) {
					pointerX = 0;
				}
				if (Math.abs(pointerY) < 0.5*(this.viewHalfY)) {
					pointerY = 0;
				}

				this.pointerX=pointerX;
				this.pointerY=pointerY;

			} else {

				let pointerX = event.pageX - this.domElement.offsetLeft - this.viewHalfX;
				let pointerY = event.pageY - this.domElement.offsetTop - this.viewHalfY;
				if (Math.abs(pointerX) < 0.2*(this.domElement.offsetLeft + this.viewHalfX)) {
					pointerX = 0;
				}
				if (Math.abs(pointerY) < 0.2*(this.domElement.offsetTop + this.viewHalfY)) {
					pointerY = 0;
				}
				this.pointerX=pointerX;
				this.pointerY=pointerY;
			}
			// console.log(this.pointerX, this.pointerY)
		};

		this.onKeyDown = function ( event ) {

			switch ( event.code ) {

				case 'ArrowUp':
				case 'KeyW': this.moveForward = true; break;

				case 'ArrowLeft':
				case 'KeyA': this.moveLeft = true; break;

				case 'ArrowDown':
				case 'KeyS': this.moveBackward = true; break;

				case 'ArrowRight':
				case 'KeyD': this.moveRight = true; break;

				case 'KeyR': this.moveUp = true; break;
				case 'KeyF': this.moveDown = true; break;

			}

		};

		this.onKeyUp = function ( event ) {

			switch ( event.code ) {

				case 'ArrowUp':
				case 'KeyW': this.moveForward = false; break;

				case 'ArrowLeft':
				case 'KeyA': this.moveLeft = false; break;

				case 'ArrowDown':
				case 'KeyS': this.moveBackward = false; break;

				case 'ArrowRight':
				case 'KeyD': this.moveRight = false; break;

				case 'KeyR': this.moveUp = false; break;
				case 'KeyF': this.moveDown = false; break;

			}

		};

		this.lookAt = function ( x, y, z ) {

			if ( x.isVector3 ) {

				_target.copy( x );

			} else {

				_target.set( x, y, z );

			}

			this.object.lookAt( _target );

			setOrientation( this );

			return this;

		};

		this.update = function () {

			const targetPosition = new Vector3();

			return function update( delta ) {

				if ( this.enabled === false ) return;

				const actualMoveSpeed = delta * this.movementSpeed;

				if ( this.moveForward || ( this.autoForward && ! this.moveBackward ) ) this.object.translateZ( - ( actualMoveSpeed ) );
				if ( this.moveBackward ) this.object.translateZ( actualMoveSpeed );

				if ( this.moveLeft ) this.object.translateX( - actualMoveSpeed );
				if ( this.moveRight ) this.object.translateX( actualMoveSpeed );

				if ( this.moveUp ) this.object.translateY( actualMoveSpeed );
				if ( this.moveDown ) this.object.translateY( - actualMoveSpeed );

				let actualLookSpeed = delta * this.lookSpeed;

				if ( ! this.activeLook ) {

					actualLookSpeed = 0;

				}

				let verticalLookRatio = 1;

				if ( this.constrainVertical ) {

					verticalLookRatio = Math.PI / ( this.verticalMax - this.verticalMin );

				}


				lon -= this.pointerX * actualLookSpeed; // pointerX > 0 --> CCW || pointerX < 0 --> CW 
				if ( this.lookVertical ) lat += this.pointerY * actualLookSpeed * verticalLookRatio;
				lat = Math.max( - 85, Math.min( 85, lat ) );

				let phi = MathUtils.degToRad( 90 + lat ); // altitude
				let theta = MathUtils.degToRad(lon);     // azimuth

				const position = this.object.position;

				// targetPosition.setFromSphericalCoords( 1, phi, theta ).add( position );
				targetPosition.x = Math.cos(theta)*Math.sin(phi);
				targetPosition.y = Math.sin(theta)*Math.sin(phi);
				targetPosition.z = Math.cos(phi);
				targetPosition.add( position );

				this.object.lookAt( targetPosition );

			};

		}();

		const _onPointerMove = this.onPointerMove.bind( this );
		const _onPointerDown = this.onPointerDown.bind( this );
		const _onPointerUp = this.onPointerUp.bind( this );
		const _onKeyDown = this.onKeyDown.bind( this );
		const _onKeyUp = this.onKeyUp.bind( this );

		this.domElement.addEventListener( 'contextmenu', contextmenu );
		this.domElement.addEventListener( 'pointerdown', _onPointerDown );
		this.domElement.addEventListener( 'pointermove', _onPointerMove );
		this.domElement.addEventListener( 'pointerup', _onPointerUp );

		window.addEventListener( 'keydown', _onKeyDown );
		window.addEventListener( 'keyup', _onKeyUp );

		function setOrientation( controls ) {

			const quaternion = controls.object.quaternion;

			_lookDirection.set( 0, 0, - 1 ).applyQuaternion( quaternion );
			_spherical.setFromVector3( _lookDirection );

			lat = 90 - MathUtils.radToDeg( _spherical.phi );
			lon = MathUtils.radToDeg( _spherical.theta );

		}

		this.handleResize();

		setOrientation( this );

	}

}

function contextmenu( event ) {

	event.preventDefault();

}

export { CameraControls };